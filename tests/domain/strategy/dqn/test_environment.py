import numpy as np
import pytest

from src.domain.strategy.dqn.environment import BatteryEnv


class TestBatteryEnv:
    @pytest.fixture
    def env(self):
        n = 24
        solar = np.array([0,0,0,0,0,0, 0.1,0.5,1.5,3.0, 4.0,4.5,4.5,4.0, 3.0,1.5,0.5,0.1, 0,0,0,0,0,0], dtype=np.float32)
        load = np.array([0.3,0.3,0.3,0.3,0.3,0.5, 0.8,1.2,1.5,1.0, 0.8,0.8,1.0,0.8, 0.8,1.0,1.5,2.0, 2.5,2.0,1.5,1.0,0.5,0.3], dtype=np.float32)
        hours = np.arange(24, dtype=np.int32)
        import_rates = np.array([0.085]*8 + [0.134]*2 + [0.182]*4 + [0.134]*4 + [0.182]*4 + [0.134]*2, dtype=np.float32)
        return BatteryEnv(
            solar=solar, load=load, hours=hours,
            import_rates=import_rates, export_rate=0.08,
            battery_capacity=15.0, max_charge_rate=4.8,
            max_discharge_rate=4.8, efficiency=0.95,
            initial_soc=0.1, min_soc_frac=0.1, dt=1.0,
        )

    def test_reset_returns_state(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) == env.state_dim

    def test_step_returns_tuple(self, env):
        env.reset()
        state, reward, done, info = env.step(4)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(4)
            steps += 1
        assert steps == 24

    def test_soc_stays_in_bounds(self, env):
        env.reset()
        for _ in range(24):
            state, _, done, _ = env.step(0)
            if done:
                break
            soc = state[0]
            assert 0.0 <= soc <= 1.0

    def test_charging_increases_soc(self, env):
        state = env.reset()
        initial_soc = state[0]
        next_state, _, _, _ = env.step(8)
        assert next_state[0] >= initial_soc

    def test_state_dim_property(self, env):
        assert env.state_dim == 16
        assert env.action_dim == 9
