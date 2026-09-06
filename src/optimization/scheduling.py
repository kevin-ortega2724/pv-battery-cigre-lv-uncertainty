from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.powerflow.cigre import build_cigre_lv
from src.powerflow.der_timeseries import synthetic_pv_profile
from src.powerflow.timeseries import normalized_load_multipliers


def tariff(index: pd.DatetimeIndex, multiplier: float = 1.0) -> np.ndarray:
    hour = np.asarray(index.hour)
    base = np.select([(hour >= 18) & (hour < 22), (hour >= 6) & (hour < 18)], [.35, .22], default=.12)
    return base * multiplier


def optimize_daily_dispatch(net_load_kw: np.ndarray, prices: np.ndarray,
                            capacity_kwh: float, power_kw: float,
                            initial_soc: float = .1, eta: float = .95,
                            demand_charge: float = .08,
                            degradation_cost: float = .01) -> dict:
    """LP schedule with daily cyclic SOC and no battery export."""
    n, dt = len(net_load_kw), .25
    # Variables: charge[n], discharge[n], energy[n+1], peak[1].
    nc, nd, ns, peak_i = 0, n, 2*n, 3*n+1
    size = peak_i + 1
    objective = np.zeros(size)
    objective[nc:nc+n] = prices * dt + degradation_cost * dt
    objective[nd:nd+n] = -prices * dt + degradation_cost * dt
    objective[peak_i] = demand_charge
    aeq, beq = [], []
    for t in range(n):
        row = np.zeros(size); row[ns+t+1] = 1; row[ns+t] = -1
        row[nc+t] = -eta*dt; row[nd+t] = dt/eta
        aeq.append(row); beq.append(0)
    aub, bub = [], []
    for t in range(n):
        # forecast net import + charge - discharge <= peak
        row = np.zeros(size); row[nc+t] = 1; row[nd+t] = -1; row[peak_i] = -1
        aub.append(row); bub.append(-net_load_kw[t])
        # discharge cannot make forecast grid import negative.
        row = np.zeros(size); row[nd+t] = 1; row[nc+t] = -1
        aub.append(row); bub.append(max(net_load_kw[t], 0))
    e0 = initial_soc * capacity_kwh
    bounds = [(0, power_kw)]*n + [(0, power_kw)]*n
    bounds += [(e0, e0)] + [(.1*capacity_kwh, .9*capacity_kwh)]*(n-1) + [(e0, e0)]
    bounds += [(0, None)]
    solved = linprog(objective, A_ub=np.asarray(aub), b_ub=np.asarray(bub),
                     A_eq=np.asarray(aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not solved.success:
        raise RuntimeError(solved.message)
    x = solved.x
    return {"charge_kw": x[:n], "discharge_kw": x[n:2*n],
            "energy_kwh": x[2*n:3*n+1], "scheduled_peak_kw": float(x[peak_i]),
            "objective": float(solved.fun)}


def make_forecast_schedules(profiles: pd.DataFrame, output_dir: str | Path,
                            scenario_count: int = 100, seed: int = 20260904) -> dict:
    net = build_cigre_lv()
    multipliers = normalized_load_multipliers(profiles, len(net.load))
    residential = net.load.index[net.load.name.str.startswith("Load R")].to_numpy()
    nominal_kw = net.load.p_mw.to_numpy() * 1000
    actual = multipliers.to_numpy()[:, residential] @ nominal_kw[residential]
    pv_capacity = .5 * nominal_kw[residential].sum()
    pv = synthetic_pv_profile(multipliers.index, seed).to_numpy() * pv_capacity
    capacity, power = 2*pv_capacity, pv_capacity
    rng = np.random.default_rng(seed)
    schedules = {"S3": [], "S4": []}; diagnostics = []
    actual_days = actual.reshape(-1, 96); pv_days = pv.reshape(-1, 96)
    residual_pool = []
    for day in range(len(actual_days)):
        idx = multipliers.index[day*96:(day+1)*96]
        if day == 0:
            zero = np.zeros(96)
            for scenario in schedules: schedules[scenario].append(pd.DataFrame({"charge_kw": zero, "discharge_kw": zero}, index=idx))
            diagnostics.append({"day": day, "status": "no_prior_day_no_dispatch"})
            continue
        point = actual_days[day-1].copy()
        if day > 1:
            residual_pool.append(actual_days[day-1] - actual_days[day-2])
        if residual_pool:
            sampled = np.stack([residual_pool[i] for i in rng.integers(0, len(residual_pool), scenario_count)])
            load_scenarios = np.clip(point + sampled, 0, None)
        else:
            load_scenarios = np.repeat(point[None, :], scenario_count, axis=0)
        q90 = np.quantile(load_scenarios, .9, axis=0)
        daily_prices = tariff(idx)
        s3 = optimize_daily_dispatch(point-pv_days[day], daily_prices, capacity, power)
        s4 = optimize_daily_dispatch(q90-pv_days[day], daily_prices, capacity, power)
        for name, solved in (("S3", s3), ("S4", s4)):
            schedules[name].append(pd.DataFrame({"charge_kw": solved["charge_kw"], "discharge_kw": solved["discharge_kw"]}, index=idx))
        diagnostics.append({"day": day, "point_mae_kw": float(np.mean(np.abs(actual_days[day]-point))),
                            "scenario_q90_mean_kw": float(q90.mean()), "s3_scheduled_peak_kw": s3["scheduled_peak_kw"],
                            "s4_scheduled_peak_kw": s4["scheduled_peak_kw"]})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    result = {"scenario_count": scenario_count, "seed": seed, "first_day_dispatch": "disabled: no prior observations",
              "forecast": "daily persistence", "uncertainty": "bootstrap of complete historical daily persistence-error blocks",
              "optimization": "daily linear program with cyclic SOC, energy tariff, peak and throughput terms"}
    for name, frames in schedules.items():
        schedule = pd.concat(frames)
        schedule.to_csv(output / f"{name.lower()}_battery_schedule.csv", index_label="timestamp")
        result[name] = {"charge_energy_kwh": float(schedule.charge_kw.sum()*.25),
                        "discharge_energy_kwh": float(schedule.discharge_kw.sum()*.25),
                        "simultaneous_intervals": int(((schedule.charge_kw>1e-8)&(schedule.discharge_kw>1e-8)).sum())}
    pd.DataFrame(diagnostics).to_csv(output / "forecast_schedule_diagnostics.csv", index=False)
    (output / "forecast_schedule_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

