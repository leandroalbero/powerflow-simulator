from datetime import datetime
from typing import Any

import pandas as pd
import pytz
from colorama import Fore, Style, init

from src.domain.battery.models import Battery
from src.domain.energy_load.model import EnergyLoad
from src.domain.energy_simulator.models import EnergySimulator
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    SelfConsumeStrategy,
)

init()


def print_header(text: str) -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_subheader(text: str) -> None:
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def format_value(value: Any, format_spec: str = "0.2f") -> str:
    if isinstance(value, (int, float)):
        return f"{value:{format_spec}}"
    return str(value)


def print_metric(label: str, value: str, unit: str = "", format_spec: str = "0.2f") -> None:
    formatted_value = format_value(value, format_spec)
    print(f"{Fore.WHITE}{label}: {Fore.GREEN}{formatted_value}{Style.RESET_ALL} {unit}")


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

    simulation_start = datetime(2024, 11, 13, 23, 00, 00, tzinfo=local_tz)
    simulation_end = datetime(2024, 11, 14, 23, 00, 00, tzinfo=local_tz)

    print_header("Energy Simulation Configuration")
    print_metric("Simulation period",
                 f"{simulation_start.strftime('%Y-%m-%d')} to {simulation_end.strftime('%Y-%m-%d')}", format_spec="s")

    solar_data = solar_data[(solar_data.index >= simulation_start) & (solar_data.index <= simulation_end)]
    load_data = load_data[(load_data.index >= simulation_start) & (load_data.index <= simulation_end)]

    solar_data = solar_data.sort_index()
    load_data = load_data.sort_index()

    print_header("Data Summary")
    print_metric("Solar data points", str(len(solar_data)), format_spec="d")
    print_metric("Load data points", str(len(load_data)), format_spec="d")
    print_metric("Solar data timespan", f"{solar_data.index.min()} to {solar_data.index.max()}", format_spec="s")
    print_metric("Load data timespan", f"{load_data.index.min()} to {load_data.index.max()}", format_spec="s")

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

    for strategy_name, strategy in strategies.items():
        print_header(f"Strategy: {strategy_name.replace('_', ' ').title()}")

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
            'solar_power': sim.solar_powers,  # Add solar power to DataFrame
            'house_consumption': sim.house_loads  # Add house consumption to DataFrame
        })

        summary_df = pd.DataFrame([results])
        file_date = simulation_start.strftime("%Y_%m")
        summary_df.to_csv(f'output_files/simulation_summary_{strategy_name}_{file_date}.csv', index=False)
        results_df.to_csv(f'output_files/simulation_detail_{strategy_name}_{file_date}.csv', index=False)

        print_subheader("Energy Metrics")
        print_metric("Total house consumption", results['total_house_consumption'], "kWh", "0.2f")
        print_metric("Total solar generated", results['total_solar_generated'], "kWh", "0.2f")
        print_metric("Total solar consumed", results['total_solar_consumed'], "kWh", "0.2f")
        print_metric("Total solar exported", results['total_solar_exported'], "kWh", "0.2f")
        print_metric("Total grid imported", results['total_grid_imported'], "kWh", "0.2f")
        print_metric("Total grid exported", results['total_solar_exported'], "kWh", "0.2f")

        print_subheader("Battery Performance")
        print_metric("Total battery in", results['total_battery_in'], "kWh", "0.2f")
        print_metric("Total battery out", results['total_battery_out'], "kWh", "0.2f")
        print_metric("Final battery level", results['battery_level'], "kWh", "0.2f")

        print_subheader("Financial & Efficiency Metrics")
        print_metric("Total cost", results['total_cost'], "€", "0.2f")

        if 'total_solar_generated' in results and results['total_solar_generated'] > 0:
            self_consumption_rate = (results['total_solar_consumed'] / results['total_solar_generated']) * 100
            print_metric("Solar self-consumption rate", self_consumption_rate, "%", "0.1f")

        if 'total_house_consumption' in results and results['total_house_consumption'] > 0:
            solar_fraction = (results['total_solar_consumed'] / results['total_house_consumption']) * 100
            print_metric("Solar fraction", solar_fraction, "%", "0.1f")

        print_subheader("Output Files")
        print(f"📊 Summary: simulation_summary_{strategy_name}_{file_date}.csv")
        print(f"📈 Details: simulation_detail_{strategy_name}_{file_date}.csv")

    print_header("Strategy Comparison")
    comparison_metrics = {
        'total_cost': ('Total Cost', '€', '0.2f'),
        'solar_fraction': ('Solar Fraction', '%', '0.1f'),
        'self_consumption_rate': ('Self-Consumption Rate', '%', '0.1f'),
        'total_solar_exported': ('Total Solar Exported', 'kWh', '0.2f')
    }

    for metric, (label, unit, format_spec) in comparison_metrics.items():
        print_subheader(label)
        for strategy_name, results in results_by_strategy.items():
            if metric in results:
                print_metric(
                    strategy_name.replace('_', ' ').title(),
                    results[metric],
                    unit,
                    format_spec
                )
