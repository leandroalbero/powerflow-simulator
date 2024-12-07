from datetime import datetime
from typing import Any, Dict
from colorama import Fore, Style


class ConsoleFormatter:
    @staticmethod
    def print_header(text: str) -> None:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")

    @staticmethod
    def print_subheader(text: str) -> None:
        print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

    @staticmethod
    def format_value(value: Any, format_spec: str = "0.2f") -> str:
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    @staticmethod
    def print_metric(label: str, value: str, unit: str = "", format_spec: str = "0.2f") -> None:
        formatted_value = ConsoleFormatter.format_value(value, format_spec)
        print(f"{Fore.WHITE}{label}: {Fore.GREEN}{formatted_value}{Style.RESET_ALL} {unit}")

    def print_simulation_config(self, start: datetime, end: datetime) -> None:
        self.print_header("Energy Simulation Configuration")
        self.print_metric("Simulation period",
                         f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
                         format_spec="s")

    def print_data_summary(self, solar_data, load_data) -> None:
        self.print_header("Data Summary")
        self.print_metric("Solar data points", str(len(solar_data)), format_spec="d")
        self.print_metric("Load data points", str(len(load_data)), format_spec="d")
        self.print_metric("Solar data timespan",
                         f"{solar_data.index.min()} to {solar_data.index.max()}",
                         format_spec="s")
        self.print_metric("Load data timespan",
                         f"{load_data.index.min()} to {load_data.index.max()}",
                         format_spec="s")

    def print_strategy_results(self, strategy_name: str, results: Dict, file_date: str) -> None:
        self.print_header(f"Strategy: {strategy_name.replace('_', ' ').title()}")

        self.print_subheader("Energy Metrics")
        self.print_metric("Total house consumption", results['total_house_consumption'], "kWh", "0.2f")
        self.print_metric("Total solar generated", results['total_solar_generated'], "kWh", "0.2f")
        self.print_metric("Total solar consumed", results['total_solar_consumed'], "kWh", "0.2f")
        self.print_metric("Total solar exported", results['total_solar_exported'], "kWh", "0.2f")
        self.print_metric("Total grid imported", results['total_grid_imported'], "kWh", "0.2f")
        self.print_metric("Total grid exported", results['total_solar_exported'], "kWh", "0.2f")

        self.print_subheader("Battery Performance")
        self.print_metric("Total battery in", results['total_battery_in'], "kWh", "0.2f")
        self.print_metric("Total battery out", results['total_battery_out'], "kWh", "0.2f")
        self.print_metric("Final battery level", results['battery_level'], "kWh", "0.2f")

        self.print_subheader("Financial & Efficiency Metrics")
        self.print_metric("Total cost", results['total_cost'], "€", "0.2f")

        if 'total_solar_generated' in results and results['total_solar_generated'] > 0:
            self_consumption_rate = (results['total_solar_consumed'] / results['total_solar_generated']) * 100
            self.print_metric("Solar self-consumption rate", self_consumption_rate, "%", "0.1f")

        if 'total_house_consumption' in results and results['total_house_consumption'] > 0:
            solar_fraction = (results['total_solar_consumed'] / results['total_house_consumption']) * 100
            self.print_metric("Solar fraction", solar_fraction, "%", "0.1f")

        self.print_subheader("Output Files")
        print(f"📊 Summary: simulation_summary_{strategy_name}_{file_date}.csv")
        print(f"📈 Details: simulation_detail_{strategy_name}_{file_date}.csv")

    def print_strategy_comparison(self, results_by_strategy: Dict) -> None:
        self.print_header("Strategy Comparison")
        comparison_metrics = {
            'total_cost': ('Total Cost', '€', '0.2f'),
            'solar_fraction': ('Solar Fraction', '%', '0.1f'),
            'self_consumption_rate': ('Self-Consumption Rate', '%', '0.1f'),
            'total_solar_exported': ('Total Solar Exported', 'kWh', '0.2f')
        }

        for metric, (label, unit, format_spec) in comparison_metrics.items():
            self.print_subheader(label)
            for strategy_name, results in results_by_strategy.items():
                if metric in results:
                    self.print_metric(
                        strategy_name.replace('_', ' ').title(),
                        results[metric],
                        unit,
                        format_spec
                    )
