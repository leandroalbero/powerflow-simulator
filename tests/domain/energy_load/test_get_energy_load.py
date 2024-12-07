from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.energy_load.model import EnergyLoad


def test_get_load_existing():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert load.get_load(timestamp) == 1.0


def test_get_load_missing():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    missing_timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert load.get_load(missing_timestamp) is None


def test_get_load_nan():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [np.nan]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert pd.isna(load.get_load(timestamp))


def test_get_load_duplicate_index():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [1.0, 2.0]
    }, index=[timestamp, timestamp])
    load = EnergyLoad(df)
    assert load.get_load(timestamp) == 1.0


def test_get_load_safe_existing():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert load.get_load_safe(timestamp) == 1.0


def test_get_load_safe_missing():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    missing_timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [1.0]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert load.get_load_safe(missing_timestamp) == 0.0
    assert load.get_load_safe(missing_timestamp, default=99.0) == 99.0


def test_get_load_safe_nan():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        'state': [np.nan]
    }, index=[timestamp])
    load = EnergyLoad(df)
    assert load.get_load_safe(timestamp) == 0.0
    assert load.get_load_safe(timestamp, default=99.0) == 99.0
