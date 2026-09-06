import numpy as np
import pandas as pd

from src.powerflow.timeseries import normalized_load_multipliers, run_s0_timeseries


def test_multipliers_use_peak_reference_and_cap_extremes():
    frame = pd.DataFrame({f"d{i}": np.arange(1, 5) * (i + 1) for i in range(15)})
    multipliers = normalized_load_multipliers(frame, 15)
    assert multipliers.max().max() <= 1.2
    assert (multipliers >= 0).all().all()


def test_short_timeseries_converges(tmp_path):
    index = pd.date_range("2000-01-01", periods=2, freq="15min")
    profiles = pd.DataFrame(np.ones((2, 15)), index=index)
    result = run_s0_timeseries(profiles, tmp_path)
    assert result["convergence_percent"] == 100
    assert result["intervals"] == 2
