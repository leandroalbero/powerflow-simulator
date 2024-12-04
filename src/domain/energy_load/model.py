from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class EnergyLoad:

    def __init__(self, load_profile: pd.DataFrame):
        if not isinstance(load_profile.index, pd.DatetimeIndex):
            raise ValueError("Load profile must have a datetime index")
        if 'state' not in load_profile.columns:
            raise ValueError("Load profile must contain a 'state' column")

        self.load_profile = load_profile

    def get_load(self, timestamp: datetime) -> Optional[float]:
        try:
            value = self.load_profile.loc[timestamp]['state']
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.iloc[0]
            if pd.isna(value):
                return np.nan
            return float(value)
        except KeyError:
            return None

    def get_load_safe(self, timestamp: datetime, default: float = 0.0) -> float:
        value = self.get_load(timestamp)
        if value is None or pd.isna(value):
            return default
        return value
