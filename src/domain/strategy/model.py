from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TypedDict

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff


@dataclass
class EnergyFlow:
    direct_solar: float = 0.0
    battery_charge: float = 0.0
    battery_discharge: float = 0.0
    grid_import: float = 0.0
    grid_export: float = 0.0
    remaining_solar: float = 0.0
    remaining_load: float = 0.0


class EnergyStrategy(ABC):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        self.battery = battery
        self.grid = grid
        self.tariff = tariff

    @abstractmethod
    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        pass


class SelfConsumeStrategy(EnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)
        self.min_battery_level = 0.1

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if flows.remaining_solar > 0:
            charge_requested = flows.remaining_solar / duration
            charged_kWh = self.battery.charge(charge_requested, duration)
            flows.battery_charge = charged_kWh
            flows.remaining_solar -= charged_kWh

            if flows.remaining_solar > 0:
                export_requested = flows.remaining_solar / duration
                exported_kWh = self.grid.export_power(export_requested, duration)
                flows.grid_export = exported_kWh
                flows.remaining_solar -= exported_kWh

        if flows.remaining_load > 0:
            battery_level = self.battery.current_charge / self.battery.capacity
            available_energy = (battery_level - self.min_battery_level) * self.battery.capacity
            max_discharge = min(
                flows.remaining_load / duration,
                self.battery.max_discharge_rate,
                available_energy / duration
            )
            discharged_kWh = self.battery.discharge(max_discharge, duration)
            flows.battery_discharge = discharged_kWh
            flows.remaining_load -= discharged_kWh

            if flows.remaining_load > 0:
                import_requested = flows.remaining_load / duration
                imported_kWh = self.grid.import_power(import_requested, duration)
                flows.grid_import = imported_kWh
                flows.remaining_load -= imported_kWh

        return flows


class ForceChargeAtNightStrategy(EnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)
        self.night_charge_power = 2.05
        self.max_charge_power = 2.05
        self.min_battery_level = 0.1
        self.min_rate = min(tariff.import_rate_schedule.values())

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()
        normalized_hour = hour % 24
        current_rate = self.tariff.get_import_rate(normalized_hour)
        is_absolute_valley = current_rate == self.min_rate

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if is_absolute_valley:
            battery_level = self.battery.current_charge / self.battery.capacity

            if battery_level < 1:
                target_charge_energy = self.night_charge_power * duration

                if flows.remaining_solar > 0:
                    max_solar_charge = min(flows.remaining_solar / duration, self.night_charge_power)
                    solar_charged = self.battery.charge(max_solar_charge, duration)
                    flows.battery_charge = solar_charged
                    flows.remaining_solar -= solar_charged * duration
                    target_charge_energy -= solar_charged * duration

                if target_charge_energy > 0:
                    charge_power_needed = target_charge_energy / duration
                    imported = self.grid.import_power(charge_power_needed, duration)
                    charged = self.battery.charge(charge_power_needed, duration)
                    flows.battery_charge += charged
                    flows.grid_import = imported

            if flows.remaining_solar > 0:
                exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                flows.grid_export = exported
                flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                import_for_load = self.grid.import_power(flows.remaining_load / duration, duration)
                flows.grid_import += import_for_load
                flows.remaining_load -= import_for_load

        else:
            if flows.remaining_solar > 0:
                solar_charged = self.battery.charge(flows.remaining_solar / duration, duration)
                flows.battery_charge = solar_charged
                flows.remaining_solar -= solar_charged * duration

                if flows.remaining_solar > 0:
                    exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                    flows.grid_export = exported
                    flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                battery_level = self.battery.current_charge / self.battery.capacity
                available_energy = (battery_level - self.min_battery_level) * self.battery.capacity
                max_discharge = min(
                    flows.remaining_load / duration,
                    self.battery.max_discharge_rate,
                    available_energy / duration
                )
                discharged = self.battery.discharge(max_discharge, duration)
                flows.battery_discharge = discharged
                flows.remaining_load -= discharged * duration

                if flows.remaining_load > 0:
                    import_req = flows.remaining_load / duration
                    imported_kWh = self.grid.import_power(import_req, duration)
                    flows.grid_import += imported_kWh
                    flows.remaining_load -= imported_kWh

        return flows


class ForceChargeAtValleyStrategy(EnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)
        self.valley_charge_target = 1.0
        self.min_battery_level = 0.1
        self.max_manual_charge_power = 2.05

        unique_rates = sorted(set(tariff.import_rate_schedule.values()))
        if len(unique_rates) >= 3:
            self.valley_rate = unique_rates[1]
            self.peak_rate = unique_rates[-1]
        else:
            raise ValueError("Tariff must have at least 3 different rates for valley strategy")

    def _is_before_peak(self, hour: int) -> bool:
        current_rate = self.tariff.get_import_rate(hour)
        next_hour = (hour + 1) % 24
        next_rate = self.tariff.get_import_rate(next_hour)
        return current_rate == self.valley_rate and next_rate == self.peak_rate

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()
        normalized_hour = hour % 24
        is_pre_peak_valley = self._is_before_peak(normalized_hour)

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if is_pre_peak_valley:
            battery_level = self.battery.current_charge / self.battery.capacity

            if battery_level < self.valley_charge_target:
                charge_needed_kWh = (self.battery.capacity * self.valley_charge_target - self.battery.current_charge)
                max_charge_per_step = min(self.max_manual_charge_power * duration,
                                          self.battery.max_charge_rate * duration)
                charge_energy = min(max_charge_per_step, charge_needed_kWh)

                if flows.remaining_solar > 0:
                    solar_charge_req = min(charge_energy, flows.remaining_solar) / duration
                    solar_charge_req = min(solar_charge_req, self.max_manual_charge_power)
                    solar_charged = self.battery.charge(solar_charge_req, duration)
                    flows.battery_charge = solar_charged
                    flows.remaining_solar -= solar_charged * duration
                    charge_energy -= solar_charged * duration

                if charge_energy > 0:
                    import_req = min(charge_energy / duration, self.max_manual_charge_power)
                    imported = self.grid.import_power(import_req, duration)
                    actual_charged = self.battery.charge(import_req, duration)
                    flows.battery_charge += actual_charged
                    flows.grid_import = imported

            if flows.remaining_solar > 0:
                exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                flows.grid_export = exported
                flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                import_for_load = self.grid.import_power(flows.remaining_load / duration, duration)
                flows.grid_import += import_for_load
                flows.remaining_load -= import_for_load

        else:
            if flows.remaining_solar > 0:
                solar_charged = self.battery.charge(flows.remaining_solar / duration, duration)
                flows.battery_charge = solar_charged
                flows.remaining_solar -= solar_charged * duration

                if flows.remaining_solar > 0:
                    exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                    flows.grid_export = exported
                    flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                current_rate = self.tariff.get_import_rate(normalized_hour)
                if current_rate == self.peak_rate:
                    battery_level = self.battery.current_charge / self.battery.capacity
                    available_energy = (battery_level - self.min_battery_level) * self.battery.capacity
                    max_discharge = min(
                        flows.remaining_load / duration,
                        self.battery.max_discharge_rate,
                        available_energy / duration
                    )
                    discharged = self.battery.discharge(max_discharge, duration)
                    flows.battery_discharge = discharged
                    flows.remaining_load -= discharged * duration

                if flows.remaining_load > 0:
                    import_req = flows.remaining_load / duration
                    imported = self.grid.import_power(import_req, duration)
                    flows.grid_import += imported
                    flows.remaining_load -= imported

        return flows


class RateBlock(TypedDict):
    start: int
    end: int
    rate: float


class ForceChargeValleyAndPrePeakStrategy(EnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)
        self.valley_charge_target = 1.0
        self.min_battery_level = 0.1
        self.max_manual_charge_power = 2.05

        unique_rates = sorted(set(tariff.import_rate_schedule.values()))
        if len(unique_rates) >= 3:
            self.valley_rate = unique_rates[0]
            self.shoulder_rate = unique_rates[1]
            self.peak_rate = unique_rates[-1]
        else:
            raise ValueError("Tariff must have at least 3 different rates for valley and pre-peak strategy")

        self.rate_blocks = self._build_rate_blocks()

    def _build_rate_blocks(self) -> List[RateBlock]:
        blocks: List[RateBlock] = []
        current_block_start: int = 0
        current_block_rate: float = 0.0

        for hour in range(25):
            if hour == 24:
                blocks.append(RateBlock(
                    start=current_block_start,
                    end=hour,
                    rate=current_block_rate
                ))
                break

            hour_rate = float(self.tariff.get_import_rate(hour))

            if hour == 0:
                current_block_rate = hour_rate
            elif hour_rate != current_block_rate:
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

    def _is_valley_block(self, block: RateBlock) -> bool:
        return block['rate'] == self.valley_rate

    def _is_peak_block(self, block: RateBlock) -> bool:
        return block['rate'] == self.peak_rate

    def _is_before_peak_block(self, block: RateBlock) -> bool:
        next_block = self._get_next_block(int(block['end'] - 1))
        return block['rate'] == self.shoulder_rate and self._is_peak_block(next_block)

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
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()
        normalized_hour = hour % 24
        should_charge = self._should_charge(normalized_hour)
        current_block = self._get_current_block(normalized_hour)

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if should_charge:
            battery_level = self.battery.current_charge / self.battery.capacity

            if battery_level < self.valley_charge_target:
                charge_needed_kWh = (self.battery.capacity * self.valley_charge_target - self.battery.current_charge)
                max_charge_per_step = min(self.max_manual_charge_power * duration,
                                          self.battery.max_charge_rate * duration)
                charge_energy = min(max_charge_per_step, charge_needed_kWh)

                if flows.remaining_solar > 0:
                    solar_charge_req = min(charge_energy, flows.remaining_solar) / duration
                    solar_charge_req = min(solar_charge_req, self.max_manual_charge_power)
                    solar_charged = self.battery.charge(solar_charge_req, duration)
                    flows.battery_charge = solar_charged
                    flows.remaining_solar -= solar_charged * duration
                    charge_energy -= solar_charged * duration

                if charge_energy > 0:
                    import_req = min(charge_energy / duration, self.max_manual_charge_power)
                    imported = self.grid.import_power(import_req, duration)
                    actual_charged = self.battery.charge(import_req, duration)
                    flows.battery_charge += actual_charged
                    flows.grid_import = imported

            if flows.remaining_solar > 0:
                exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                flows.grid_export = exported
                flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                import_for_load = self.grid.import_power(flows.remaining_load / duration, duration)
                flows.grid_import += import_for_load
                flows.remaining_load -= import_for_load

        else:
            if flows.remaining_solar > 0:
                solar_charged = self.battery.charge(flows.remaining_solar / duration, duration)
                flows.battery_charge = solar_charged
                flows.remaining_solar -= solar_charged * duration

                if flows.remaining_solar > 0:
                    exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                    flows.grid_export = exported
                    flows.remaining_solar -= exported

            if flows.remaining_load > 0:
                if self._is_peak_block(current_block):
                    battery_level = self.battery.current_charge / self.battery.capacity
                    available_energy = (battery_level - self.min_battery_level) * self.battery.capacity
                    max_discharge = min(
                        flows.remaining_load / duration,
                        self.battery.max_discharge_rate,
                        available_energy / duration
                    )
                    discharged = self.battery.discharge(max_discharge, duration)
                    flows.battery_discharge = discharged
                    flows.remaining_load -= discharged * duration

                if flows.remaining_load > 0:
                    import_req = flows.remaining_load / duration
                    imported = self.grid.import_power(import_req, duration)
                    flows.grid_import += imported
                    flows.remaining_load -= imported

        return flows
