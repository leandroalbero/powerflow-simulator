import pandas as pd
import pytest
import pytz

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn_strategy import DqnStrategy
from src.domain.strategy.model import EnergyFlow

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

    def test_missing_policy_raises_clear_error(self, battery, grid, tariff, tmp_path, monkeypatch):
        from src.domain.strategy import dqn_strategy

        monkeypatch.setattr(dqn_strategy, "REPO_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="DQN policy not trained: output_files/dqn_policy.pt missing"):
            dqn_strategy.load_dqn_strategy(battery, grid, tariff)

    def test_loads_trained_policy_from_repo_relative_path(self, battery, grid, tariff, agent, tmp_path, monkeypatch):
        from src.domain.strategy import dqn_strategy

        (tmp_path / "output_files").mkdir()
        agent.save(str(tmp_path / "output_files" / "dqn_policy.pt"))
        monkeypatch.setattr(dqn_strategy, "REPO_ROOT", tmp_path)
        strategy = dqn_strategy.load_dqn_strategy(battery, grid, tariff, min_battery_level=0.2)
        assert strategy.min_battery_level == 0.2
        strategy.set_timestamp(pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ))
        assert isinstance(strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0), EnergyFlow)
