from datetime import datetime
from typing import Dict, Optional

from src.domain.battery.models import Battery
from src.domain.energy_load.model import EnergyLoad
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.strategy.model import (
    EnergyFlow,
    EnergyStrategy,
    SelfConsumeStrategy,
)


class EnergySimulator:
    total_cost: float
    total_solar_generated: float
    total_solar_consumed: float
    total_solar_exported: float
    total_grid_imported: float
    total_battery_in: float
    total_battery_out: float
    total_house_consumption: float
    timestamps: list
    battery_levels: list
    grid_imports: list
    grid_exports: list
    solar_powers: list  # New list for solar power values
    house_loads: list   # New list for house consumption values

    def __init__(self, battery: Battery, load: EnergyLoad, grid: Grid,
                 tariff: PowerTariff, solar: SolarGenerator,
                 strategy: Optional[EnergyStrategy] = None):
        self.battery = battery
        self.load = load
        self.grid = grid
        self.tariff = tariff
        self.solar = solar
        self.strategy = strategy or SelfConsumeStrategy(battery, grid, tariff)
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.total_cost = 0.0
        self.total_solar_generated = 0.0
        self.total_solar_consumed = 0.0
        self.total_solar_exported = 0.0
        self.total_grid_imported = 0.0
        self.total_battery_in = 0.0
        self.total_battery_out = 0.0
        self.total_house_consumption = 0.0
        self.timestamps = []
        self.battery_levels = []
        self.grid_imports = []
        self.grid_exports = []
        self.solar_powers = []    # Initialize new list
        self.house_loads = []     # Initialize new list

    def step(self, timestamp: datetime, prev_timestamp: Optional[datetime] = None) -> None:
        if not timestamp.tzinfo:
            raise ValueError("Timestamp must be timezone-aware")

        hour = timestamp.hour

        if prev_timestamp is not None:
            duration = (timestamp - prev_timestamp).total_seconds() / 3600.0
        else:
            duration = 1.0 / 60.0

        solar_power = self.solar.get_generation_safe(timestamp, default=0.0) / 1000.0
        load_power = self.load.get_load_safe(timestamp, default=0.0) / 1000.0

        solar_energy = solar_power * duration
        self.total_solar_generated += solar_energy
        self.total_house_consumption += load_power * duration

        flows = self.strategy.calculate_energy_flows(solar_power, load_power, hour, duration)

        self._update_metrics(flows, hour, duration)

        self.timestamps.append(timestamp)
        self.battery_levels.append(self.battery.current_charge)
        self.grid_imports.append(flows.grid_import)
        self.grid_exports.append(flows.grid_export)
        self.solar_powers.append(solar_power)      # Store solar power
        self.house_loads.append(load_power)        # Store house consumption

    def _update_metrics(self, flows: EnergyFlow, hour: int, duration: float) -> None:
        self.total_solar_consumed += flows.direct_solar
        self.total_battery_in += flows.battery_charge * duration
        self.total_battery_out += flows.battery_discharge * duration

        self.total_grid_imported += flows.grid_import * duration
        self.total_solar_exported += flows.grid_export * duration

        self.total_cost += flows.grid_import * duration * self.tariff.get_import_rate(hour)
        self.total_cost -= flows.grid_export * duration * self.tariff.get_export_rate(hour)

    def get_metrics(self) -> Dict:
        metrics = {'total_cost': round(self.total_cost, 2),
                   'total_solar_generated': round(self.total_solar_generated, 2),
                   'total_solar_consumed': round(self.total_solar_consumed, 2),
                   'total_solar_exported': round(self.total_solar_exported, 2),
                   'total_grid_imported': round(self.total_grid_imported, 2),
                   'total_battery_in': round(self.total_battery_in, 2),
                   'total_battery_out': round(self.total_battery_out, 2),
                   'total_house_consumption': round(self.total_house_consumption, 2),
                   'battery_level': round(self.battery.current_charge, 2),
                   'battery_capacity': round(self.battery.capacity, 2)}

        # Always include total_house_consumption even if it's 0

        # Calculate rates only if denominators are positive
        if self.total_solar_generated > 0:
            metrics['self_consumption_rate'] = round(
                (self.total_solar_consumed / self.total_solar_generated) * 100, 1)
        else:
            metrics['self_consumption_rate'] = 0.0

        if self.total_house_consumption > 0:
            metrics['solar_fraction'] = round(
                (self.total_solar_consumed / self.total_house_consumption) * 100, 1)
        else:
            metrics['solar_fraction'] = 0.0

        return metrics
