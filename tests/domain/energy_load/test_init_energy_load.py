import pytest
import pandas as pd

from src.domain.energy_load.model import EnergyLoad


def test_init_valid():
    df = pd.DataFrame({
        'state': [1.0, 2.0, 3.0]
    }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
    load = EnergyLoad(df)
    assert load.load_profile.equals(df)


def test_init_invalid_index():
    df = pd.DataFrame({
        'state': [1.0, 2.0, 3.0]
    })
    with pytest.raises(ValueError, match="Load profile must have a datetime index"):
        EnergyLoad(df)


def test_init_missing_state_column():
    df = pd.DataFrame({
        'value': [1.0, 2.0, 3.0]
    }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
    with pytest.raises(ValueError, match="Load profile must contain a 'state' column"):
        EnergyLoad(df)
