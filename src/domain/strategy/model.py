from abc import ABC, abstractmethod
from dataclasses import dataclass

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
            discharge_requested = flows.remaining_load / duration
            discharged_kWh = self.battery.discharge(discharge_requested, duration)
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
        self.off_peak_hours = {0, 1, 2, 3, 4, 5, 6, 7}
        self.peak_hours = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        self.min_rate = min(tariff.import_rate_schedule.values())
        self.max_rate = max(tariff.import_rate_schedule.values())
        self.night_charge_power = 0.8
        self.max_charge_power = 2.05
        self.day_discharge_hours = {9, 10, 11, 12, 13}
        self.min_battery_level = 0.1  # Keep 10% minimum charge

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()
        normalized_hour = hour % 24

        battery_level = self.battery.current_charge / self.battery.capacity
        is_off_peak = normalized_hour in self.off_peak_hours
        is_peak = normalized_hour in self.peak_hours

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if is_off_peak and battery_level < 1:
            target_charge_energy = self.night_charge_power * duration
            if flows.remaining_solar > 0:
                solar_charged = self.battery.charge(flows.remaining_solar / duration, duration)
                flows.battery_charge = solar_charged
                flows.remaining_solar -= solar_charged * duration
                target_charge_energy -= solar_charged * duration

            if target_charge_energy > 0:
                import_charged = self.battery.charge(target_charge_energy / duration, duration)
                flows.battery_charge += import_charged
                flows.grid_import = import_charged

            if flows.remaining_solar > 0:
                exported = self.grid.export_power(flows.remaining_solar / duration, duration)
                flows.grid_export = exported
                flows.remaining_solar -= exported

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
            if is_peak:
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
                imported = self.grid.import_power(flows.remaining_load / duration, duration)
                flows.grid_import += imported
                flows.remaining_load -= imported

        return flows

class ForceChargeAtValleyStrategy(EnergyStrategy):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff):
        super().__init__(battery, grid, tariff)

        rates = list(tariff.import_rate_schedule.values())
        self.min_import_rate = min(rates)
        self.max_import_rate = max(rates)

        self.valley_rate_threshold = 0.1340
        self.valley_charge_target = 0.95
        self.min_battery_level = 0.1

        self.valley_hours = {
            h for (start, end), rate in tariff.import_rate_schedule.items()
            for h in range(start, end)
            if rate <= self.valley_rate_threshold
        }

        peak_rate_threshold = self.max_import_rate * 0.8
        self.peak_hours = {
            h for (start, end), rate in tariff.import_rate_schedule.items()
            for h in range(start, end)
            if rate >= peak_rate_threshold
        }

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        flows = EnergyFlow()

        battery_level_percent = self.battery.current_charge / self.battery.capacity
        available_battery_discharge = max(
            0, self.battery.current_charge - (self.battery.capacity * self.min_battery_level)
        )

        normalized_hour = hour % 24

        current_rate = next(
            (rate for (start, end), rate in self.tariff.import_rate_schedule.items()
             if start <= normalized_hour < end),
            self.max_import_rate
        )

        is_valley = normalized_hour in self.valley_hours
        is_peak = normalized_hour in self.peak_hours

        if is_valley and battery_level_percent < self.valley_charge_target:
            charge_needed_kWh = (self.battery.capacity * self.valley_charge_target - self.battery.current_charge)
            max_charge_per_step = self.battery.max_charge_rate * duration
            charge_energy = min(max_charge_per_step, charge_needed_kWh)

            if charge_energy > 0:
                total_import_needed = load_energy + charge_energy
                import_req = total_import_needed / duration
                imported = self.grid.import_power(import_req, duration)
                flows.grid_import = imported

                charge_req = charge_energy / duration
                actual_charged = self.battery.charge(charge_req, duration)
                flows.battery_charge = actual_charged

                remaining_import_for_load = imported - actual_charged

                flows.direct_solar = min(solar_energy, load_energy)
                flows.remaining_solar = solar_energy - flows.direct_solar

                load_after_solar = load_energy - flows.direct_solar
                load_after_import = load_after_solar - remaining_import_for_load
                flows.remaining_load = max(0, load_after_import)
                return flows

        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        if flows.remaining_solar > 0:
            if is_valley:
                export_req = flows.remaining_solar / duration
                exported = self.grid.export_power(export_req, duration)
                flows.grid_export = exported
                flows.remaining_solar -= exported

                if flows.remaining_solar > 0:
                    charge_req = flows.remaining_solar / duration
                    charged = self.battery.charge(charge_req, duration)
                    flows.battery_charge = charged
                    flows.remaining_solar -= charged
            else:
                charge_req = flows.remaining_solar / duration
                charged = self.battery.charge(charge_req, duration)
                flows.battery_charge = charged
                flows.remaining_solar -= charged

                if flows.remaining_solar > 0:
                    export_req = flows.remaining_solar / duration
                    exported = self.grid.export_power(export_req, duration)
                    flows.grid_export = exported
                    flows.remaining_solar -= exported

        if flows.remaining_load > 0:
            if available_battery_discharge > 0 and (is_peak or current_rate > self.valley_rate_threshold):
                discharge_req = min(flows.remaining_load, available_battery_discharge) / duration
                discharged = self.battery.discharge(discharge_req, duration)
                flows.battery_discharge = discharged
                flows.remaining_load -= discharged

                if flows.remaining_load > 0:
                    import_req = flows.remaining_load / duration
                    imported = self.grid.import_power(import_req, duration)
                    flows.grid_import = imported
                    flows.remaining_load -= imported
            else:
                import_req = flows.remaining_load / duration
                imported = self.grid.import_power(import_req, duration)
                flows.grid_import = imported
                flows.remaining_load -= imported

                if flows.remaining_load > 0 and available_battery_discharge > 0:
                    discharge_req = min(flows.remaining_load, available_battery_discharge) / duration
                    discharged = self.battery.discharge(discharge_req, duration)
                    flows.battery_discharge = discharged
                    flows.remaining_load -= discharged

        return flows
