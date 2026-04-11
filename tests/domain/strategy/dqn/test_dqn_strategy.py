import numpy as np
import pytest
import pytz
import pandas as pd

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn_strategy import DqnStrategy

LOCAL_TZ = pytz.timezone("Europe/Madrid")


class TestDqnStrategy:
    @pytest.fixture
    def tariff(self):
        schedule = {
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
        }
        return PowerTariff(rate_schedule=schedule)

    @pytest.fixture
    def battery(self):
        return Battery(capacity=15.0, max_charge_rate=4.8, max_discharge_rate=4.8, efficiency=0.95)

    @pytest.fixture
    def grid(self):
        return Grid(max_import=5.0, max_export=5.0)

    @pytest.fixture
    def agent(self):
        return DqnAgent(state_dim=16, action_dim=9)

    def test_returns_valid_flow(self, battery, grid, tariff, agent):
        strategy = DqnStrategy(battery=battery, grid=grid, tariff=tariff, agent=agent)
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)

    def test_with_saved_model(self, battery, grid, tariff, agent, tmp_path):
        path = tmp_path / "test.pt"
        agent.save(str(path))
        loaded_agent = DqnAgent(state_dim=16, action_dim=9)
        loaded_agent.load(str(path))
        strategy = DqnStrategy(battery=battery, grid=grid, tariff=tariff, agent=loaded_agent)
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)
