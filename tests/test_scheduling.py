import numpy as np
from src.optimization.scheduling import optimize_daily_dispatch


def test_daily_lp_respects_energy_bounds_cycle_and_no_simultaneous_action():
    load = np.r_[np.full(48, 20.), np.full(48, 60.)]
    prices = np.r_[np.full(48, .1), np.full(48, .3)]
    result = optimize_daily_dispatch(load, prices, 100, 50)
    assert abs(result["energy_kwh"][0] - result["energy_kwh"][-1]) < 1e-7
    assert result["energy_kwh"].min() >= 10-1e-7 and result["energy_kwh"].max() <= 90+1e-7
    assert not np.any((result["charge_kw"] > 1e-7) & (result["discharge_kw"] > 1e-7))
