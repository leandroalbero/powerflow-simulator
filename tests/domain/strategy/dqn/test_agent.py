import numpy as np
import pytest
import torch

from src.domain.strategy.dqn.agent import DqnAgent


class TestDqnAgent:
    @pytest.fixture
    def agent(self):
        return DqnAgent(state_dim=16, action_dim=9)

    def test_select_action_returns_valid_index(self, agent):
        state = np.random.randn(16).astype(np.float32)
        action = agent.select_action(state, epsilon=0.0)
        assert 0 <= action < 9

    def test_select_action_with_full_exploration(self, agent):
        state = np.random.randn(16).astype(np.float32)
        actions = [agent.select_action(state, epsilon=1.0) for _ in range(100)]
        assert len(set(actions)) > 1

    def test_store_and_sample(self, agent):
        for _ in range(100):
            s = np.random.randn(16).astype(np.float32)
            a = np.random.randint(9)
            r = np.random.randn()
            s2 = np.random.randn(16).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)
        batch = agent.sample_batch(32)
        assert batch is not None
        assert batch[0].shape == (32, 16)

    def test_train_step_runs(self, agent):
        for _ in range(64):
            s = np.random.randn(16).astype(np.float32)
            a = np.random.randint(9)
            r = np.random.randn()
            s2 = np.random.randn(16).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)
        loss = agent.train_step()
        assert loss is not None
        assert loss >= 0

    def test_save_and_load(self, agent, tmp_path):
        state = np.random.randn(16).astype(np.float32)
        q_before = agent.get_q_values(state)
        path = tmp_path / "test_model.pt"
        agent.save(str(path))
        agent2 = DqnAgent(state_dim=16, action_dim=9)
        agent2.load(str(path))
        q_after = agent2.get_q_values(state)
        np.testing.assert_array_almost_equal(q_before, q_after)
