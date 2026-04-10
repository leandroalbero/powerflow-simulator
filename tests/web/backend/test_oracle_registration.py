import pytest

from web.backend.services.simulation_service import STRATEGY_MAP, _STRATEGY_CLASSES


class TestOracleRegistration:
    def test_oracle_in_strategy_registry(self):
        """Oracle should appear in the strategy registry."""
        assert "oracle" in STRATEGY_MAP
        assert STRATEGY_MAP["oracle"].name == "Oracle Optimizer"

    def test_oracle_in_strategy_classes(self):
        """Oracle strategy class should be registered."""
        assert "oracle" in _STRATEGY_CLASSES
