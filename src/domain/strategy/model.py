from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TypedDict, Optional, Tuple
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection, RateType


@dataclass
class EnergyFlow:
    direct_solar: float = 0.0
    battery_charge: float = 0.0
    battery_discharge: float = 0.0
    grid_import: float = 0.0
    grid_export: float = 0.0
    remaining_solar: float = 0.0
    remaining_load: float = 0.0


class BaseEnergyStrategy(ABC):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        self.battery = battery
        self.grid = grid
        self.tariff = tariff
        self.min_battery_level = 0.1
        self.max_charge_power = 2.05

    @abstractmethod
    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")
        pass

    def _get_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        return self.tariff.get_rate(hour % 24, direction)

    def _calculate_initial_flows(self, solar_energy: float, load_energy: float) -> EnergyFlow:
        flows = EnergyFlow()
        solar_energy = max(0.0, solar_energy)
        load_energy = max(0.0, load_energy)

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar
        return flows

    def _handle_remaining_solar(self, flows: EnergyFlow, duration: float) -> None:
        if flows.remaining_solar > 0:
            solar_charged = float(self.battery.charge(flows.remaining_solar / duration, duration))
            flows.battery_charge = solar_charged
            flows.remaining_solar -= solar_charged * duration

            if flows.remaining_solar > 0:
                exported = float(self.grid.export_power(flows.remaining_solar / duration, duration))
                flows.grid_export = exported
                flows.remaining_solar -= exported * duration

    def _handle_remaining_load(self, flows: EnergyFlow, duration: float,
                               force_discharge: bool = False) -> None:
        if flows.remaining_load > 0:
            if force_discharge:
                discharged = float(self._discharge_battery(flows.remaining_load, duration))
                flows.battery_discharge = discharged
                flows.remaining_load -= discharged * duration

            if flows.remaining_load > 0:
                import_req = flows.remaining_load / duration
                imported = float(self.grid.import_power(import_req, duration))
                flows.grid_import += imported
                flows.remaining_load -= imported * duration

    def _discharge_battery(self, energy_needed: float, duration: float) -> float:
        battery_level = self.battery.current_charge / self.battery.capacity
        available_energy = (battery_level - self.min_battery_level) * self.battery.capacity
        max_discharge = min(
            energy_needed / duration,
            self.battery.max_discharge_rate,
            available_energy / duration
        )
        return float(self.battery.discharge(max_discharge, duration))

    def _charge_battery(self, target_charge_energy: float, duration: float) -> float:
        import_req = min(target_charge_energy / duration, self.max_charge_power)
        imported = float(self.grid.import_power(import_req, duration))
        actual_charged = float(self.battery.charge(import_req, duration))
        return actual_charged


class SelfConsumeStrategy(BaseEnergyStrategy):
    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
        self._handle_remaining_solar(flows, duration)
        self._handle_remaining_load(flows, duration, force_discharge=True)
        return flows


class RateBlock(TypedDict):
    start: int
    end: int
    rate: Rate


class BaseRateAwareStrategy(BaseEnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)
        self.valley_charge_target = 1.0
        self.rate_blocks = self._build_rate_blocks()
        self._initialize_rates()

    def _initialize_rates(self) -> None:
        import_rates = sorted(
            [rate for rate in self.tariff.rate_schedule.values()
             if rate.energy_direction == EnergyDirection.IMPORT],
            key=lambda x: x.price
        )

        if len(import_rates) < 3:
            raise ValueError("Tariff must have at least 3 different rates for this strategy")

        self.valley_rate = import_rates[0]
        self.shoulder_rate = import_rates[1]
        self.peak_rate = import_rates[-1]

    def _build_rate_blocks(self) -> List[RateBlock]:
        blocks: List[RateBlock] = []
        current_block_start: int = 0
        current_block_rate: Optional[Rate] = None

        for hour in range(25):
            if hour == 24:
                if current_block_rate:
                    blocks.append(RateBlock(
                        start=current_block_start,
                        end=hour,
                        rate=current_block_rate
                    ))
                break

            hour_rate = self._get_rate(hour, EnergyDirection.IMPORT)

            if hour == 0:
                current_block_rate = hour_rate
            elif (hour_rate.price != current_block_rate.price or
                  hour_rate.rate_type != current_block_rate.rate_type):
                blocks.append(RateBlock(
                    start=current_block_start,
                    end=hour,
                    rate=current_block_rate
                ))
                current_block_start = hour
                current_block_rate = hour_rate

        return blocks

    def _get_current_block(self, hour: int) -> RateBlock:
        normalized_hour = hour % 24
        for block in self.rate_blocks:
            if block['start'] <= normalized_hour < block['end']:
                return block
        return self.rate_blocks[-1]

    def _get_next_block(self, hour: int) -> RateBlock:
        current_block = self._get_current_block(hour)
        for i, block in enumerate(self.rate_blocks):
            if block == current_block:
                return self.rate_blocks[(i + 1) % len(self.rate_blocks)]
        return self.rate_blocks[0]

    def _is_peak_block(self, block: RateBlock) -> bool:
        return (block['rate'].price == self.peak_rate.price and
                block['rate'].rate_type == self.peak_rate.rate_type)

    def _handle_charging_period(self, flows: EnergyFlow, duration: float) -> None:
        battery_level = self.battery.current_charge / self.battery.capacity

        if battery_level < self.valley_charge_target:
            charge_needed_kWh = (
                self.battery.capacity * self.valley_charge_target - self.battery.current_charge)
            max_charge_per_step = min(
                self.max_charge_power * duration,
                self.battery.max_charge_rate * duration
            )
            charge_energy = min(max_charge_per_step, charge_needed_kWh)
            actual_charged = self._charge_battery(charge_energy, duration)
            flows.battery_charge += actual_charged
            flows.grid_import += actual_charged

        if flows.remaining_solar > 0:
            exported = float(self.grid.export_power(flows.remaining_solar / duration, duration))
            flows.grid_export = exported
            flows.remaining_solar -= exported * duration

        self._handle_remaining_load(flows, duration)


class ForceChargeAtNightStrategy(BaseRateAwareStrategy):
    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
        current_rate = self._get_rate(hour % 24, EnergyDirection.IMPORT)
        is_valley = (current_rate.price == self.valley_rate.price and
                     current_rate.rate_type == RateType.VALLEY)

        if is_valley:
            self._handle_remaining_solar(flows, duration)
            battery_level = self.battery.current_charge / self.battery.capacity

            if battery_level < self.valley_charge_target:
                charge_needed_kwh = (self.battery.capacity * self.valley_charge_target -
                                     self.battery.current_charge)
                max_charge_per_step = min(
                    self.max_charge_power * duration,
                    self.battery.max_charge_rate * duration
                )
                charge_energy = min(max_charge_per_step, charge_needed_kwh)

                actual_charged = self._charge_battery(charge_energy, duration)
                flows.battery_charge += actual_charged
                flows.grid_import += actual_charged

            self._handle_remaining_load(flows, duration)
        else:
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration, force_discharge=True)

        return flows

class ForceChargeAtValleyStrategy(BaseRateAwareStrategy):
    def _is_before_peak(self, hour: int) -> bool:
        current_rate = self._get_rate(hour, EnergyDirection.IMPORT)
        next_rate = self._get_rate((hour + 1) % 24, EnergyDirection.IMPORT)
        return (current_rate.price == self.shoulder_rate.price and
                next_rate.price == self.peak_rate.price)

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)

        if hour < 0 or hour >= 24:
            actual_charged = self._charge_battery(flows.remaining_solar, duration)
            flows.battery_charge += actual_charged
            flows.grid_import += actual_charged
            return flows

        is_pre_peak_valley = self._is_before_peak(hour % 24)

        if is_pre_peak_valley:
            self._handle_charging_period(flows, duration)
        else:
            self._handle_remaining_solar(flows, duration)
            current_block = self._get_current_block(hour % 24)
            force_discharge = self._is_peak_block(current_block)
            self._handle_remaining_load(flows, duration, force_discharge=force_discharge)

        return flows


class ForceChargeValleyAndPrePeakStrategy(BaseRateAwareStrategy):
    def _is_valley_block(self, block: RateBlock) -> bool:
        return (block['rate'].price == self.valley_rate.price and
                block['rate'].rate_type == self.valley_rate.rate_type)

    def _is_before_peak_block(self, block: RateBlock) -> bool:
        next_block = self._get_next_block(int(block['end'] - 1))
        return (block['rate'].price == self.shoulder_rate.price and
                self._is_peak_block(next_block))

    def _is_after_valley_block(self, hour: int) -> bool:
        current_block = self._get_current_block(hour)
        for i, block in enumerate(self.rate_blocks):
            if block == current_block:
                prev_block = self.rate_blocks[i - 1] if i > 0 else self.rate_blocks[-1]
                return self._is_valley_block(prev_block)
        return False

    def _should_charge(self, hour: int) -> bool:
        current_block = self._get_current_block(hour)
        normalized_hour = hour % 24

        if self._is_after_valley_block(normalized_hour):
            return False

        if self._is_valley_block(current_block):
            next_block = self._get_next_block(normalized_hour)
            return not self._is_valley_block(next_block) or self.battery.charge_level < self.valley_charge_target

        return self._is_before_peak_block(current_block)

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)

        if hour < 0 or hour >= 24:
            actual_charged = self._charge_battery(flows.remaining_solar, duration)
            flows.battery_charge += actual_charged
            flows.grid_import += actual_charged
            return flows

        normalized_hour = hour % 24
        should_charge = self._should_charge(normalized_hour)
        current_block = self._get_current_block(normalized_hour)

        if should_charge:
            self._handle_charging_period(flows, duration)
        else:
            self._handle_remaining_solar(flows, duration)
            force_discharge = self._is_peak_block(current_block)
            self._handle_remaining_load(flows, duration, force_discharge=force_discharge)

        return flows
