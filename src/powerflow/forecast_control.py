from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import pandapower as pp

from .cigre import build_cigre_lv
from .der_timeseries import _summary, synthetic_pv_profile
from .timeseries import normalized_load_multipliers


def run_forecast_control(profiles: pd.DataFrame, schedule: pd.DataFrame,
                         output_dir: str | Path, scenario: str,
                         pv_penetration: float = .5, seed: int = 20260904) -> dict:
    net = build_cigre_lv()
    multipliers = normalized_load_multipliers(profiles, len(net.load))
    schedule = schedule.reindex(multipliers.index)
    if schedule.isna().any().any():
        raise ValueError("Schedule and profiles are not aligned")
    p_nom, q_nom = net.load.p_mw.to_numpy(copy=True), net.load.q_mvar.to_numpy(copy=True)
    residential = net.load.index[net.load.name.str.startswith("Load R")].to_numpy()
    residential_kw = p_nom[residential] * 1000
    pv_total_kw = pv_penetration * residential_kw.sum()
    shares = residential_kw / residential_kw.sum()
    pv_kw = pv_total_kw * shares
    pv = synthetic_pv_profile(multipliers.index, seed)
    sgens = [pp.create_sgen(net, int(net.load.at[i, "bus"]), p_mw=0, q_mvar=0,
                            name=f"PV_BESS_{i}") for i in residential]
    capacity_kwh = 2 * pv_total_kw
    soc_kwh = .1 * capacity_kwh
    rows = []
    for step, (timestamp, multiplier) in enumerate(multipliers.iterrows()):
        scale = multiplier.to_numpy()
        net.load.loc[:, "p_mw"] = p_nom * scale
        net.load.loc[:, "q_mvar"] = q_nom * scale
        pv_now = pv_kw * pv.iloc[step]
        charge_total = float(schedule.iloc[step].charge_kw)
        discharge_total = float(schedule.iloc[step].discharge_kw)
        charge, discharge = shares * charge_total, shares * discharge_total
        soc_kwh += .95 * charge_total * .25 - discharge_total * .25 / .95
        net.sgen.loc[sgens, "p_mw"] = (pv_now + discharge - charge) / 1000
        try:
            pp.runpp(net, algorithm="nr", init="flat" if step == 0 else "results",
                     tolerance_mva=1e-8, max_iteration=50, calculate_voltage_angles=True, numba=False)
            vm = net.res_bus.vm_pu
            rows.append({"timestamp": timestamp, "converged": True,
                         "minimum_voltage_pu": vm.min(), "maximum_voltage_pu": vm.max(),
                         "undervoltage_bus_count": int((vm < .95).sum()), "overvoltage_bus_count": int((vm > 1.05).sum()),
                         "maximum_line_loading_percent": net.res_line.loading_percent.max(),
                         "maximum_transformer_loading_percent": net.res_trafo.loading_percent.max(),
                         "active_line_losses_kw": net.res_line.pl_mw.sum()*1000,
                         "grid_import_kw": net.res_ext_grid.p_mw.sum()*1000,
                         "pv_generation_kw": pv_now.sum(), "battery_charge_kw": charge_total,
                         "battery_discharge_kw": discharge_total, "curtailed_pv_kw": 0,
                         "mean_soc": soc_kwh/capacity_kwh})
        except pp.LoadflowNotConverged:
            rows.append({"timestamp": timestamp, "converged": False})
    frame = pd.DataFrame(rows).set_index("timestamp")
    summary = _summary(frame, scenario)
    summary.update({"pv_penetration_percent": 100*pv_penetration, "pv_installed_kw": pv_total_kw,
                    "forecast_control": "S3 daily-persistence point forecast" if scenario == "S3" else "S4 q90 envelope from 100 block-bootstrap error scenarios",
                    "daily_terminal_soc_constraint": True})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"cigre_{scenario.lower()}_timeseries.csv")
    (output / f"cigre_{scenario.lower()}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
