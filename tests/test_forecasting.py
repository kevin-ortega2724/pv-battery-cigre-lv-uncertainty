import numpy as np
import pandas as pd

from src.forecasting.features import make_supervised
from src.forecasting.metrics import interval_metrics, pinball


def test_feature_target_alignment_and_no_current_load_feature():
    y = pd.Series(np.arange(800.0), index=pd.date_range("2008-01-01", periods=800, freq="15min"))
    frame = make_supervised(y, horizon_steps=4)
    first = frame.iloc[0]
    issue_position = y.index.get_loc(frame.index[0])
    assert first["target"] == y.iloc[issue_position + 4]
    assert first["last_observed"] == y.iloc[issue_position]
    assert first["lag_1"] == y.iloc[issue_position - 1]
    assert "current_load" not in frame.columns


def test_probabilistic_metrics_for_perfect_interval():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball(y, y, 0.5) == 0
    assert interval_metrics(y, y - 1, y + 1)["picp_percent"] == 100


def test_day_ahead_uses_issue_observation_without_lag_zero_column():
    y = pd.Series(np.arange(900.0), index=pd.date_range("2008-01-01", periods=900, freq="15min"))
    frame = make_supervised(y, horizon_steps=96)
    assert "last_observed" in frame
    assert "lag_0" not in frame
