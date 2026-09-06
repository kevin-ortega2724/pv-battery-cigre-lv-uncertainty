from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import pandapower as pp

from .cigre import build_cigre_lv


def normalized_load_multipliers(profiles: pd.DataFrame, n_loads: int,
                                reference_quantile: float = .99,
                                maximum_multiplier: float = 1.2) -> pd.DataFrame:
    """Map shapes to loads with the benchmark demand treated as a peak reference."""
    if profiles.shape[1] < n_loads:
        raise ValueError(f"Need at least {n_loads} profiles, got {profiles.shape[1]}")
    selected = profiles.iloc[:, :n_loads].astype(float)
    reference = selected.quantile(reference_quantile, axis=0)
    if (reference <= 0).any():
        raise ValueError("Every profile must have a positive mean")
    values = selected.divide(reference, axis=1).clip(upper=maximum_multiplier)
    values.columns = range(n_loads)
    return values


def summarize_timeseries(timeseries: pd.DataFrame) -> dict:
    dt_hours = .25
    valid = timeseries.loc[timeseries["converged"].astype(bool)]
    worst_voltage_time = valid["minimum_voltage_pu"].idxmin()
    peak_import_time = valid["grid_import_kw"].idxmax()
    return {
        "scenario": "S0 demand without distributed generation",
        "profile_interpretation": "Synthetic dwelling-derived shapes normalized by their 99th percentile and capped at 1.2 as temporal proxy multipliers. CIGRE tabular P is treated as a peak-reference level; load categories and Q/P ratios are preserved.",
        "intervals": int(len(timeseries)), "days": float(len(timeseries) / 96),
        "convergence_percent": float(100 * timeseries["converged"].astype(bool).mean()),
        "algorithm_counts": {str(k): int(v) for k, v in timeseries["algorithm"].value_counts().items()},
        "minimum_voltage_pu": float(valid.minimum_voltage_pu.min()),
        "minimum_voltage_timestamp": str(worst_voltage_time),
        "p01_minimum_voltage_pu": float(valid.minimum_voltage_pu.quantile(.01)),
        "intervals_with_undervoltage": int((valid.undervoltage_bus_count > 0).sum()),
        "undervoltage_interval_percent": float(100 * (valid.undervoltage_bus_count > 0).mean()),
        "intervals_with_overvoltage": int((valid.overvoltage_bus_count > 0).sum()),
        "maximum_line_loading_percent": float(valid.maximum_line_loading_percent.max()),
        "intervals_with_line_overload": int((valid.maximum_line_loading_percent > 100).sum()),
        "maximum_transformer_loading_percent": float(valid.maximum_transformer_loading_percent.max()),
        "intervals_with_transformer_overload": int((valid.maximum_transformer_loading_percent > 100).sum()),
        "peak_grid_import_kw": float(valid.grid_import_kw.max()),
        "peak_grid_import_timestamp": str(peak_import_time),
        "mean_grid_import_kw": float(valid.grid_import_kw.mean()),
        "active_line_loss_energy_kwh": float(valid.active_line_losses_kw.sum(min_count=1) * dt_hours),
        "mean_active_line_losses_kw": float(valid.active_line_losses_kw.mean()),
        "loss_energy_percent_of_import": float(100 * valid.active_line_losses_kw.sum(min_count=1) / valid.grid_import_kw.sum(min_count=1)),
    }


def run_s0_timeseries(profiles: pd.DataFrame, output_dir: str | Path,
                      voltage_limits=(0.95, 1.05)) -> dict:
    net = build_cigre_lv()
    p_nominal = net.load.p_mw.to_numpy(copy=True)
    q_nominal = net.load.q_mvar.to_numpy(copy=True)
    multipliers = normalized_load_multipliers(profiles, len(net.load))
    time_rows, voltage_rows = [], []
    for step, (timestamp, multiplier) in enumerate(multipliers.iterrows()):
        scale = multiplier.to_numpy()
        net.load.loc[:, "p_mw"] = p_nominal * scale
        net.load.loc[:, "q_mvar"] = q_nominal * scale
        algorithm = "nr"
        try:
            pp.runpp(net, algorithm="nr", init="flat" if step == 0 else "results", tolerance_mva=1e-8,
                     max_iteration=50, calculate_voltage_angles=True, numba=False)
        except pp.LoadflowNotConverged:
            algorithm = "bfsw"
            try:
                pp.runpp(net, algorithm="bfsw", init="flat", tolerance_mva=1e-8,
                         max_iteration=200, calculate_voltage_angles=True, numba=False)
            except pp.LoadflowNotConverged:
                time_rows.append({"timestamp": timestamp, "converged": False,
                                  "algorithm": "failed", "minimum_voltage_pu": np.nan,
                                  "maximum_voltage_pu": np.nan, "undervoltage_bus_count": np.nan,
                                  "overvoltage_bus_count": np.nan, "active_line_losses_kw": np.nan,
                                  "reactive_line_losses_kvar": np.nan, "maximum_line_loading_percent": np.nan,
                                  "maximum_transformer_loading_percent": np.nan, "grid_import_kw": np.nan,
                                  "grid_import_kvar": np.nan})
                voltage_rows.append(np.full(len(net.bus), np.nan))
                continue
        vm = net.res_bus.vm_pu.copy()
        time_rows.append({
            "timestamp": timestamp,
            "converged": bool(net.converged),
            "algorithm": algorithm,
            "minimum_voltage_pu": float(vm.min()),
            "maximum_voltage_pu": float(vm.max()),
            "undervoltage_bus_count": int((vm < voltage_limits[0]).sum()),
            "overvoltage_bus_count": int((vm > voltage_limits[1]).sum()),
            "active_line_losses_kw": float(net.res_line.pl_mw.sum() * 1000),
            "reactive_line_losses_kvar": float(net.res_line.ql_mvar.sum() * 1000),
            "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
            "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
            "grid_import_kw": float(net.res_ext_grid.p_mw.sum() * 1000),
            "grid_import_kvar": float(net.res_ext_grid.q_mvar.sum() * 1000),
        })
        voltage_rows.append(vm.to_numpy())
    timeseries = pd.DataFrame(time_rows).set_index("timestamp")
    voltages = pd.DataFrame(voltage_rows, index=timeseries.index,
                            columns=[f"bus_{i}" for i in net.bus.index])
    summary = summarize_timeseries(timeseries)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(output / "cigre_s0_timeseries.csv")
    voltages.to_csv(output / "cigre_s0_bus_voltages.csv")
    (output / "cigre_s0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
