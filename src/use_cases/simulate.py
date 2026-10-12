"""Headless comparison of the basic strategies on the bundled high-resolution CSVs."""

from datetime import datetime

import pandas as pd
import pytz

from src.data.frame import build_power_frame, csv_to_entity_frame, memory_reader
from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import run
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    NoBatteryStrategy,
    SelfConsumeStrategy,
)
from src.use_cases.print_console import ConsoleFormatter

STRATEGIES = {
    'no_battery': NoBatteryStrategy,
    'self_consume': SelfConsumeStrategy,
    'charge_night': ForceChargeAtNightStrategy,
    'force_valleys': ForceChargeAtValleyStrategy,
    'force_valleys_pre_peak': ForceChargeValleyAndPrePeakStrategy,
}


def build_tariff() -> PowerTariff:
    return PowerTariff(
        rate_schedule={
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
        },
        weekend_rate=Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
    )


if __name__ == '__main__':
    local_tz = pytz.timezone('Europe/Madrid')
    start, end = '2024-01-05', '2024-12-05'
    entities = {
        'pv': csv_to_entity_frame('data/pv_power_highres.csv'),
        'load': csv_to_entity_frame('data/house_consumption_highres.csv'),
    }
    frame = build_power_frame(start, end, reader=memory_reader(entities))

    formatter = ConsoleFormatter()
    formatter.print_simulation_config(local_tz.localize(datetime(2024, 1, 5)), local_tz.localize(datetime(2024, 12, 5)))
    valid = frame[frame['valid']]
    formatter.print_data_summary(valid[['pv_kw']], valid[['load_kw']])

    results_by_strategy = {}
    file_date = start[:7].replace('-', '_')
    for name, cls in STRATEGIES.items():
        tariff = build_tariff()
        battery = Battery(capacity=5.4, max_charge_rate=2.1, max_discharge_rate=2.1, initial_soc=0.1)
        strategy = cls(battery, Grid(max_import=5.0, max_export=5.0), tariff)
        result = run(frame, strategy, tariff)
        metrics = {k: v for k, v in result.metrics.items() if k != 'monthly'}
        results_by_strategy[name] = metrics

        detail = pd.DataFrame(result.timeseries)
        detail['ts'] = pd.to_datetime(detail['ts'], unit='ms', utc=True).dt.tz_convert(local_tz)
        pd.DataFrame([metrics]).to_csv(f'output_files/simulation_summary_{name}_{file_date}.csv', index=False)
        detail.to_csv(f'output_files/simulation_detail_{name}_{file_date}.csv', index=False)

        formatter.print_strategy_results(name, metrics, file_date)

    formatter.print_strategy_comparison(results_by_strategy)
