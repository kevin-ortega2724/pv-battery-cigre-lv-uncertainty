import pandas as pd
from src.powerflow.der_timeseries import synthetic_pv_profile


def test_synthetic_pv_is_seeded_bounded_and_zero_at_night():
    index = pd.date_range("2000-01-01", periods=96, freq="15min")
    a = synthetic_pv_profile(index, seed=3)
    b = synthetic_pv_profile(index, seed=3)
    pd.testing.assert_series_equal(a, b)
    assert a.between(0, 1).all()
    assert a.loc[index.hour < 6].eq(0).all()
    assert a.loc[index.hour >= 18].eq(0).all()
