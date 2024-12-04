from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class SolarGenerator:

    def __init__(self, generation_profile: pd.DataFrame):
        if not isinstance(generation_profile.index, pd.DatetimeIndex):
            raise ValueError("Generation profile must have a datetime index")
        if 'state' not in generation_profile.columns:
            raise ValueError("Generation profile must contain a 'state' column")

        self.generation_profile = generation_profile

    def get_generation(self, timestamp: datetime) -> Optional[float]:
        try:
            value = self.generation_profile.loc[timestamp]['state']
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.iloc[0]
            if pd.isna(value):
                return np.nan
            return float(value)
        except KeyError:
            return None

    def get_generation_safe(self, timestamp: datetime, default: float = 0.0) -> float:
        value = self.get_generation(timestamp)
        if value is None or pd.isna(value):
            return default
        return value
