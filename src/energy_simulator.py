import pandas as pd
import numpy as np


class EnergySimulator:
    def __init__(self, pv_file, export_power_file, battery_soc_file, battery_power_file, battery_capacity, debug=False):
        self.pv_file = pv_file
        self.export_power_file = export_power_file
        self.battery_soc_file = battery_soc_file
        self.battery_power_file = battery_power_file
        self.battery_capacity = battery_capacity  # in kWh
        self.df = None
        self.statistics = {}
        self.debug = debug

    def load_data(self, simulate_date):
        if self.debug:
            print("Loading data from CSV files...")
        pv_df = pd.read_csv(self.pv_file)
        export_power_df = pd.read_csv(self.export_power_file)
        battery_soc_df = pd.read_csv(self.battery_soc_file)
        battery_power_df = pd.read_csv(self.battery_power_file)

        if self.debug:
            print("Parsing datetime columns and converting 'state' columns to numeric...")
        pv_df['last_changed'] = pd.to_datetime(pv_df['last_changed'], utc=True)
        export_power_df['last_changed'] = pd.to_datetime(export_power_df['last_changed'], utc=True)
        battery_soc_df['last_changed'] = pd.to_datetime(battery_soc_df['last_changed'], utc=True)
        battery_power_df['last_changed'] = pd.to_datetime(battery_power_df['last_changed'], utc=True)

        pv_df['state'] = pd.to_numeric(pv_df['state'], errors='coerce')
        export_power_df['state'] = pd.to_numeric(export_power_df['state'], errors='coerce')
        battery_soc_df['state'] = pd.to_numeric(battery_soc_df['state'], errors='coerce')
        battery_power_df['state'] = pd.to_numeric(battery_power_df['state'], errors='coerce')

        start_date = pd.to_datetime(simulate_date).tz_localize('UTC')
        end_date = start_date + pd.Timedelta(days=1)

        if self.debug:
            print(f"Filtering data between {start_date} and {end_date}...")
        pv_df = pv_df[(pv_df['last_changed'] >= start_date) & (pv_df['last_changed'] < end_date)]
        export_power_df = export_power_df[
            (export_power_df['last_changed'] >= start_date) & (export_power_df['last_changed'] < end_date)]
        battery_soc_df = battery_soc_df[
            (battery_soc_df['last_changed'] >= start_date) & (battery_soc_df['last_changed'] < end_date)]
        battery_power_df = battery_power_df[
            (battery_power_df['last_changed'] >= start_date) & (battery_power_df['last_changed'] < end_date)]

        if self.debug:
            print("Resampling data to hourly intervals...")
        pv_df.set_index('last_changed', inplace=True)
        export_power_df.set_index('last_changed', inplace=True)
        battery_soc_df.set_index('last_changed', inplace=True)
        battery_power_df.set_index('last_changed', inplace=True)

        pv_df = pv_df.resample('1h').ffill()
        export_power_df = export_power_df.resample('1h').ffill()
        battery_soc_df = battery_soc_df.resample('1h').ffill()
        battery_power_df = battery_power_df.resample('1h').ffill()

        pv_df.reset_index(inplace=True)
        export_power_df.reset_index(inplace=True)
        battery_soc_df.reset_index(inplace=True)
        battery_power_df.reset_index(inplace=True)

        if self.debug:
            print("Merging dataframes...")
        df = pd.merge(pv_df, export_power_df, on='last_changed', how='outer', suffixes=('_pv', '_export'))
        df = pd.merge(df, battery_soc_df, on='last_changed', how='outer')
        df = pd.merge(df, battery_power_df, on='last_changed', how='outer', suffixes=('', '_battery_power'))
        df.rename(columns={'state': 'SoC', 'state_battery_power': 'battery_power'}, inplace=True)

        df = df[['last_changed', 'state_pv', 'state_export', 'battery_power', 'SoC']]
        df[['state_pv', 'state_export', 'battery_power', 'SoC']] = df[
            ['state_pv', 'state_export', 'battery_power', 'SoC']].ffill()
        df.dropna(inplace=True)
        df.sort_values('last_changed', inplace=True)
        df.reset_index(drop=True, inplace=True)

        if self.debug:
            print("Data loading and preprocessing complete.")

        self.df = df

    def simulate(self, simulate_date):
        if self.debug:
            print("Starting simulation...")
        self.load_data(simulate_date)
        df = self.df.copy()

        if self.debug:
            print("Calculating energy flows and battery operations...")
        df['energy_pv'] = df['state_pv'] / 1000.0  # Convert W to kW
        df['energy_export'] = df['state_export'] / 1000.0  # Convert W to kW
        df['energy_battery'] = df['battery_power'] / 1000.0  # Convert W to kW

        # Compute energy consumption
        df['energy_consumption'] = df['energy_pv'] - df['energy_export'] + df['energy_battery']

        # Calculate grid import and export
        df['grid_out'] = df['energy_export'].clip(lower=0)
        df['grid_in'] = (-df['energy_export']).clip(lower=0)

        # Battery charging and discharging
        df['battery_charging'] = df['energy_battery'].clip(lower=0)
        df['battery_discharging'] = (-df['energy_battery']).clip(lower=0)

        # Energy from PV to battery
        df['energy_pv_to_battery'] = df[['battery_charging', 'energy_pv']].min(axis=1)
        df['energy_grid_to_battery'] = df['battery_charging'] - df['energy_pv_to_battery']

        # Timestamps
        df['hour'] = df['last_changed'].dt.hour
        df['weekday'] = df['last_changed'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5

        if self.debug:
            print("Calculating costs and revenues...")
        weekday_conditions = [
            (df['is_weekend'] == False) & (df['hour'] >= 0) & (df['hour'] < 8),
            (df['is_weekend'] == False) & (df['hour'] >= 8) & (df['hour'] < 10),
            (df['is_weekend'] == False) & (df['hour'] >= 10) & (df['hour'] < 14),
            (df['is_weekend'] == False) & (df['hour'] >= 14) & (df['hour'] < 18),
            (df['is_weekend'] == False) & (df['hour'] >= 18) & (df['hour'] < 22),
            (df['is_weekend'] == False) & (df['hour'] >= 22) & (df['hour'] < 24),
            (df['is_weekend'] == True)
        ]

        rates = [0.0850, 0.1340, 0.1820, 0.1340, 0.1820, 0.1340, 0.0850]
        df['rate'] = np.select(weekday_conditions, rates)

        df['cost'] = df['grid_in'] * df['rate']
        df['revenue'] = df['grid_out'] * 0.08

        # Calculate total solar production
        total_solar_production = df['energy_pv'].sum()

        total_grid_in = df['grid_in'].sum()
        total_grid_out = df['grid_out'].sum()
        total_energy_pv_to_battery = df['energy_pv_to_battery'].sum()
        total_energy_grid_to_battery = df['energy_grid_to_battery'].sum()
        total_house_consumption = df['energy_consumption'].sum()
        total_battery_out_to_house = df['battery_discharging'].sum()
        total_cost = df['cost'].sum()
        total_revenue = df['revenue'].sum()
        net_cost = total_cost - total_revenue

        self.statistics = {
            'Total Solar Production (kWh)': total_solar_production,
            'Grid IN (kWh)': total_grid_in,
            'Grid OUT (kWh)': total_grid_out,
            'Battery charging from PV (kWh)': total_energy_pv_to_battery,
            'Battery charging from Grid (kWh)': total_energy_grid_to_battery,
            'Total House Consumption (kWh)': total_house_consumption,
            'Battery Discharging to House (kWh)': total_battery_out_to_house,
            'Total Cost (€)': total_cost,
            'Total Revenue (€)': total_revenue,
            'Net Cost (€)': net_cost
        }

        if self.debug:
            print("Simulation complete. Statistics calculated:")
            for key, value in self.statistics.items():
                print(f"{key}: {value:.2f}")

    def get_statistics(self):
        return self.statistics
