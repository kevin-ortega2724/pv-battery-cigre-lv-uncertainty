from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import pandapower as pp

from .cigre import build_cigre_lv
from .timeseries import normalized_load_multipliers


def synthetic_pv_profile(index: pd.DatetimeIndex, seed: int = 20260904) -> pd.Series:
    """Physics-inspired normalized PV availability; explicitly not measured weather."""
    hour = np.asarray(index.hour + index.minute / 60, dtype=float)
    clear = np.sin(np.pi * (hour - 6) / 12)
    clear = np.clip(clear, 0, None)
    clear[clear < 1e-12] = 0
    rng = np.random.default_rng(seed)
    unique_days = pd.Index(index.normalize().unique())
    cloud = np.clip(rng.beta(8, 2, size=len(unique_days)), .45, 1.0)
    cloud_by_day = pd.Series(cloud, index=unique_days)
    availability = clear * cloud_by_day.loc[index.normalize()].to_numpy()
    return pd.Series(availability, index=index, name="pv_availability_pu")


def _summary(frame: pd.DataFrame, scenario: str) -> dict:
    valid = frame.loc[frame.converged]
    return {
        "scenario": scenario, "intervals": int(len(frame)),
        "convergence_percent": float(100 * frame.converged.mean()),
        "minimum_voltage_pu": float(valid.minimum_voltage_pu.min()),
        "undervoltage_intervals": int((valid.undervoltage_bus_count > 0).sum()),
        "undervoltage_interval_percent": float(100 * (valid.undervoltage_bus_count > 0).mean()),
        "overvoltage_intervals": int((valid.overvoltage_bus_count > 0).sum()),
        "maximum_line_loading_percent": float(valid.maximum_line_loading_percent.max()),
        "maximum_transformer_loading_percent": float(valid.maximum_transformer_loading_percent.max()),
        "transformer_overload_intervals": int((valid.maximum_transformer_loading_percent > 100).sum()),
        "peak_grid_import_kw": float(valid.grid_import_kw.max()),
        "peak_grid_export_kw": float(np.maximum(-valid.grid_import_kw.min(), 0)),
        "import_energy_kwh": float(np.clip(valid.grid_import_kw, 0, None).sum() * .25),
        "export_energy_kwh": float(np.clip(-valid.grid_import_kw, 0, None).sum() * .25),
        "active_line_loss_energy_kwh": float(valid.active_line_losses_kw.sum() * .25),
        "pv_energy_kwh": float(valid.pv_generation_kw.sum() * .25),
        "battery_charge_energy_kwh": float(valid.battery_charge_kw.sum() * .25),
        "battery_discharge_energy_kwh": float(valid.battery_discharge_kw.sum() * .25),
        "curtailed_pv_energy_kwh": float(valid.curtailed_pv_kw.sum() * .25),
        "ending_mean_soc": float(valid.mean_soc.iloc[-1]) if valid.mean_soc.notna().any() else None,
    }


def run_der_scenario(profiles: pd.DataFrame, output_dir: str | Path, scenario: str,
                     pv_penetration: float = .5, seed: int = 20260904) -> dict:
    if scenario not in {"S1", "S2"}:
        raise ValueError("scenario must be S1 or S2")
    net = build_cigre_lv()
    p_nom, q_nom = net.load.p_mw.to_numpy(copy=True), net.load.q_mvar.to_numpy(copy=True)
    multipliers = normalized_load_multipliers(profiles, len(net.load))
    residential = net.load.index[net.load.name.str.startswith("Load R")].to_numpy()
    residential_kw = p_nom[residential] * 1000
    pv_total_kw = pv_penetration * residential_kw.sum()
    pv_kw = pv_total_kw * residential_kw / residential_kw.sum()
    pv = synthetic_pv_profile(multipliers.index, seed)
    sgen_indices = [pp.create_sgen(net, int(net.load.at[i, "bus"]), p_mw=0, q_mvar=0,
                                   name=f"PV_{i}") for i in residential]
    capacity_kwh = 2 * pv_kw
    power_kw = pv_kw.copy()
    soc = .1 * capacity_kwh
    rows = []
    for step, (timestamp, multiplier) in enumerate(multipliers.iterrows()):
        scale = multiplier.to_numpy()
        load_kw = p_nom * scale * 1000
        net.load.loc[:, "p_mw"] = load_kw / 1000
        net.load.loc[:, "q_mvar"] = q_nom * scale
        pv_now = pv_kw * pv.iloc[step]
        charge = np.zeros(len(residential)); discharge = np.zeros(len(residential))
        if scenario == "S2":
            local_load = load_kw[residential]
            surplus = np.maximum(pv_now - local_load, 0)
            room_input = np.maximum(.9 * capacity_kwh - soc, 0) / (.95 * .25)
            charge = np.minimum.reduce([surplus, power_kw, room_input])
            if 18 <= timestamp.hour < 22:
                available_energy_output = np.maximum(soc - .1 * capacity_kwh, 0) * .95
                remaining_hours = 22 - (timestamp.hour + timestamp.minute / 60)
                sustainable_output = available_energy_output / max(remaining_hours, .25)
                discharge = np.minimum.reduce([local_load, power_kw, sustainable_output])
            soc += .95 * charge * .25 - discharge * .25 / .95
        net.sgen.loc[sgen_indices, "p_mw"] = (pv_now + discharge - charge) / 1000
        try:
            pp.runpp(net, algorithm="nr", init="flat" if step == 0 else "results",
                     tolerance_mva=1e-8, max_iteration=50, calculate_voltage_angles=True, numba=False)
            vm = net.res_bus.vm_pu
            rows.append({"timestamp": timestamp, "converged": True,
                         "minimum_voltage_pu": vm.min(), "maximum_voltage_pu": vm.max(),
                         "undervoltage_bus_count": int((vm < .95).sum()),
                         "overvoltage_bus_count": int((vm > 1.05).sum()),
                         "maximum_line_loading_percent": net.res_line.loading_percent.max(),
                         "maximum_transformer_loading_percent": net.res_trafo.loading_percent.max(),
                         "active_line_losses_kw": net.res_line.pl_mw.sum() * 1000,
                         "grid_import_kw": net.res_ext_grid.p_mw.sum() * 1000,
                         "pv_generation_kw": pv_now.sum(), "battery_charge_kw": charge.sum(),
                         "battery_discharge_kw": discharge.sum(), "curtailed_pv_kw": 0,
                         "mean_soc": float(np.mean(soc / capacity_kwh)) if scenario == "S2" else np.nan,
                         "minimum_soc": float(np.min(soc / capacity_kwh)) if scenario == "S2" else np.nan,
                         "maximum_soc": float(np.max(soc / capacity_kwh)) if scenario == "S2" else np.nan})
        except pp.LoadflowNotConverged:
            rows.append({"timestamp": timestamp, "converged": False})
    frame = pd.DataFrame(rows).set_index("timestamp")
    summary = _summary(frame, scenario)
    summary.update({"pv_penetration_percent": 100 * pv_penetration,
                    "pv_installed_kw": float(pv_total_kw),
                    "pv_profile_provenance": "Synthetic clear-sky-shaped availability with seeded daily beta cloud factor; not measured irradiance.",
                    "battery_model": None if scenario == "S1" else {"duration_hours": 2, "eta_charge": .95, "eta_discharge": .95, "soc_limits": [.1,.9], "initial_soc": .1, "rule": "charge local PV surplus; spread available discharge over remaining 18:00-22:00 window"}})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"cigre_{scenario.lower()}_timeseries.csv")
    (output / f"cigre_{scenario.lower()}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"load_index": residential, "bus": net.load.loc[residential, "bus"].to_numpy(),
                  "pv_kw": pv_kw, "battery_kwh": capacity_kwh if scenario == "S2" else 0,
                  "battery_kw": power_kw if scenario == "S2" else 0}).to_csv(output / f"cigre_{scenario.lower()}_der_placement.csv", index=False)
    return summary
