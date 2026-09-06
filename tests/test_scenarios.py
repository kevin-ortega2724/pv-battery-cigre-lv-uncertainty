import numpy as np
import pandas as pd

from src.scenarios.dwellings import generate_dwellings, validation_metrics


def test_synthetic_profiles_are_nonnegative_reproducible_and_diverse():
    index = pd.date_range("2010-01-01", periods=96 * 8, freq="15min")
    values = 1 + .5 * np.sin(2 * np.pi * np.arange(len(index)) / 96)
    measured = pd.Series(values, index=index)
    a, meta_a = generate_dwellings(measured, n_dwellings=4, n_days=3, seed=7)
    b, meta_b = generate_dwellings(measured, n_dwellings=4, n_days=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(meta_a, meta_b)
    assert (a >= 0).all().all()
    assert not a.iloc[:, 0].equals(a.iloc[:, 1])
    metrics = validation_metrics(a)
    assert 0 < metrics["peak_coincidence_factor"] <= 1
