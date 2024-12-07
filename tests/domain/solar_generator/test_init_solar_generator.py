import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.domain.solar_generator.solar_generator import SolarGenerator


def test_init_valid_data():
    df = pd.DataFrame({
        'state': [1.0, 2.0, 3.0]
    }, index=pd.date_range('2024-01-01', periods=3))
    generator = SolarGenerator(df)
    assert generator.generation_profile.equals(df)


def test_init_invalid_index():
    df = pd.DataFrame({
        'state': [1.0, 2.0, 3.0]
    })
    with pytest.raises(ValueError, match="Generation profile must have a datetime index"):
        SolarGenerator(df)


def test_init_missing_state_column():
    df = pd.DataFrame({
        'other': [1.0, 2.0, 3.0]
    }, index=pd.date_range('2024-01-01', periods=3))
    with pytest.raises(ValueError, match="Generation profile must contain a 'state' column"):
        SolarGenerator(df)


def test_get_generation_exact_match():
    df = pd.DataFrame({
        'state': [1.0, 2.0, 3.0]
    }, index=pd.date_range('2024-01-01', periods=3))
    generator = SolarGenerator(df)
    assert generator.get_generation(datetime(2024, 1, 1)) == 1.0


def test_get_generation_missing_timestamp():
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert generator.get_generation(datetime(2024, 1, 2)) is None


def test_get_generation_nan_value():
    df = pd.DataFrame({
        'state': [np.nan]
    }, index=[datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert pd.isna(generator.get_generation(datetime(2024, 1, 1)))


def test_get_generation_duplicate_index():
    df = pd.DataFrame({
        'state': [1.0, 2.0]
    }, index=[datetime(2024, 1, 1), datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert generator.get_generation(datetime(2024, 1, 1)) == 1.0


def test_get_generation_safe_valid_value():
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert generator.get_generation_safe(datetime(2024, 1, 1)) == 1.0


def test_get_generation_safe_missing_timestamp():
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert generator.get_generation_safe(datetime(2024, 1, 2)) == 0.0
    assert generator.get_generation_safe(datetime(2024, 1, 2), default=5.0) == 5.0


def test_get_generation_safe_nan_value():
    df = pd.DataFrame({
        'state': [np.nan]
    }, index=[datetime(2024, 1, 1)])
    generator = SolarGenerator(df)
    assert generator.get_generation_safe(datetime(2024, 1, 1)) == 0.0
    assert generator.get_generation_safe(datetime(2024, 1, 1), default=5.0) == 5.0
