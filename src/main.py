from colorama import init, Fore, Style

from src.energy_simulator import EnergySimulator


def main():
    init(autoreset=True)
    battery_capacity = 5  # kWh

    # Enable or disable debug mode
    debug_mode = False  # Set False to disable debug output

    simulator = EnergySimulator(
        'data/pv_power.csv',
        'data/export_power.csv',
        'data/battery_soc.csv',
        'data/battery_power.csv',
        battery_capacity,
        debug=debug_mode
    )

    simulate_date = '2024-06-01'
    simulator.simulate(simulate_date)
    stats = simulator.get_statistics()

    print(f"{Fore.CYAN}{Style.BRIGHT}Energy Simulation Results for {simulate_date}{Style.RESET_ALL}\n")

    print(f"{Fore.GREEN}{Style.BRIGHT}Energy Metrics:{Style.RESET_ALL}")
    energy_metrics = [
        'Total Solar Production (kWh)',
        'Grid IN (kWh)',
        'Grid OUT (kWh)',
        'Battery charging from PV (kWh)',
        'Battery charging from Grid (kWh)',
        'Total House Consumption (kWh)',
        'Battery Discharging to House (kWh)'
    ]
    for key in energy_metrics:
        value = stats[key]
        print(f"{Fore.GREEN}{key}: {value:.2f} kWh{Style.RESET_ALL}")

    print(f"\n{Fore.BLUE}{Style.BRIGHT}Financial Metrics:{Style.RESET_ALL}")
    financial_metrics = [
        'Total Cost (€)',
        'Total Revenue (€)',
        'Net Cost (€)'
    ]
    for key in financial_metrics:
        value = stats[key]
        print(f"{Fore.BLUE}{key}: €{value:.2f}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
