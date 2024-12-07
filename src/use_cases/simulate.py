from datetime import datetime

import pandas as pd
import pytz

from src.domain.battery.models import Battery
from src.domain.energy_load.model import EnergyLoad
from src.domain.energy_simulator.models import EnergySimulator
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    SelfConsumeStrategy,
)
from src.use_cases.print_console import ConsoleFormatter

if __name__ == '__main__':
    solar_data = pd.read_csv(
        'data/pv_power_highres.csv',
        index_col='last_changed',
    )

    load_data = pd.read_csv(
        'data/house_consumption_highres.csv',
        index_col='last_changed',
    )

    local_tz = pytz.timezone('Europe/Madrid')
    solar_data.index = pd.to_datetime(solar_data.index, utc=True).tz_convert(local_tz)
    load_data.index = pd.to_datetime(load_data.index, utc=True).tz_convert(local_tz)

    solar_data = solar_data.tz_convert(local_tz)
    load_data = load_data.tz_convert(local_tz)

    simulation_start = datetime(2024, 12, 1, 23, 00, 00, tzinfo=local_tz)
    simulation_end = datetime(2024, 12, 5, 23, 00, 00, tzinfo=local_tz)

    formatter = ConsoleFormatter()
    formatter.print_simulation_config(simulation_start, simulation_end)

    solar_data = solar_data[(solar_data.index >= simulation_start) & (solar_data.index <= simulation_end)]
    load_data = load_data[(load_data.index >= simulation_start) & (load_data.index <= simulation_end)]

    solar_data = solar_data.sort_index()
    load_data = load_data.sort_index()

    formatter.print_data_summary(solar_data, load_data)

    tariff = PowerTariff(
        rate_schedule={
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT)
        }
    )

    battery = Battery(capacity=5.4, max_charge_rate=2.1, max_discharge_rate=2.1)
    load = EnergyLoad(load_data)
    grid = Grid(max_import=5.0, max_export=5.0)
    solar = SolarGenerator(solar_data)

    self_consume_strategy = SelfConsumeStrategy(battery, grid, tariff)
    charge_night_strategy = ForceChargeAtNightStrategy(battery, grid, tariff)
    force_charging_at_valleys_strategy = ForceChargeAtValleyStrategy(battery, grid, tariff)
    force_charging_at_valleys_and_pre_peak_strategy = ForceChargeValleyAndPrePeakStrategy(battery, grid, tariff)

    strategies = {
        'self_consume': self_consume_strategy,
        'charge_night': charge_night_strategy,
        'force_valleys': force_charging_at_valleys_strategy,
        'force_valleys_pre_peak': force_charging_at_valleys_and_pre_peak_strategy
    }

    results_by_strategy = {}
    file_date = simulation_start.strftime("%Y_%m")

    for strategy_name, strategy in strategies.items():
        sim = EnergySimulator(battery, load, grid, tariff, solar, strategy=strategy)
        battery.current_charge = 0.54

        prev_timestamp = None
        for timestamp in load_data.index:
            sim.step(timestamp, prev_timestamp)
            prev_timestamp = timestamp

        results = sim.get_metrics()
        results_by_strategy[strategy_name] = results

        results_df = pd.DataFrame({
            'timestamp': sim.timestamps,
            'battery_level': sim.battery_levels,
            'grid_import': sim.grid_imports,
            'grid_export': sim.grid_exports,
            'solar_power': sim.solar_powers,
            'house_consumption': sim.house_loads
        })

        summary_df = pd.DataFrame([results])
        summary_df.to_csv(f'output_files/simulation_summary_{strategy_name}_{file_date}.csv', index=False)
        results_df.to_csv(f'output_files/simulation_detail_{strategy_name}_{file_date}.csv', index=False)

        formatter.print_strategy_results(strategy_name, results, file_date)

    formatter.print_strategy_comparison(results_by_strategy)
