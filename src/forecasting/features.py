from __future__ import annotations

import numpy as np
import pandas as pd


def make_supervised(series: pd.Series, horizon_steps: int) -> pd.DataFrame:
    """Create issue-time features for a direct h-step forecast.

    Every load-derived feature uses data available at or before issue time t.
    Calendar features describe target time t+h and are known in advance.
    """
    y = series.astype(float).sort_index()
    target_time = y.index + pd.Timedelta(minutes=15 * horizon_steps)
    out = pd.DataFrame(index=y.index)
    out["target"] = y.shift(-horizon_steps)
    # Values at issue time and earlier are observable. The target-aligned daily
    # and weekly lags are 96-h and 672-h steps behind the issue time.
    out["last_observed"] = y
    for lag in sorted({1, 4, 96, 672, 96 - horizon_steps, 672 - horizon_steps}):
        if lag <= 0:
            continue
        out[f"lag_{lag}"] = y.shift(lag)
    history = y
    for window in (4, 16, 96, 672):
        roll = history.rolling(window, min_periods=window)
        out[f"mean_{window}"] = roll.mean()
        out[f"std_{window}"] = roll.std()
        out[f"min_{window}"] = roll.min()
        out[f"max_{window}"] = roll.max()
    minute_of_day = target_time.hour * 60 + target_time.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["dow_sin"] = np.sin(2 * np.pi * target_time.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * target_time.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (target_time.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (target_time.month - 1) / 12)
    out["weekend"] = (target_time.dayofweek >= 5).astype(int)
    return out.dropna()


def temporal_split(frame: pd.DataFrame):
    train = frame.loc[:"2008-12-31 23:59:59"]
    validation = frame.loc["2009-01-01":"2009-12-31 23:59:59"]
    test = frame.loc["2010-01-01":"2010-11-30 23:59:59"]
    return train, validation, test
