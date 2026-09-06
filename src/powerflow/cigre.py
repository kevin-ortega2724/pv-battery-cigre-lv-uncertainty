from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import pandapower as pp
from pandapower.networks import create_cigre_network_lv


def build_cigre_lv():
    """Return pandapower's documented CIGRE LV benchmark without DER modifications."""
    return create_cigre_network_lv()


def _summary(net) -> dict:
    return {
        "converged": bool(net.converged),
        "buses": int(len(net.bus)),
        "lines": int(len(net.line)),
        "transformers": int(len(net.trafo)),
        "loads": int(len(net.load)),
        "minimum_voltage_pu": float(net.res_bus.vm_pu.min()),
        "maximum_voltage_pu": float(net.res_bus.vm_pu.max()),
        "active_line_losses_kw": float(net.res_line.pl_mw.sum() * 1000),
        "reactive_line_losses_kvar": float(net.res_line.ql_mvar.sum() * 1000),
        "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
        "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
        "grid_import_kw": float(net.res_ext_grid.p_mw.sum() * 1000),
    }


def compare_algorithms(tolerance_mva: float = 1e-9, max_iteration: int = 100) -> dict:
    base = build_cigre_lv()
    nr, bfsw = copy.deepcopy(base), copy.deepcopy(base)
    pp.runpp(nr, algorithm="nr", tolerance_mva=tolerance_mva, max_iteration=max_iteration, calculate_voltage_angles=True, numba=False)
    pp.runpp(bfsw, algorithm="bfsw", tolerance_mva=tolerance_mva, max_iteration=max_iteration, calculate_voltage_angles=True, numba=False)
    common = nr.res_bus.index.intersection(bfsw.res_bus.index)
    dv = np.abs(nr.res_bus.loc[common, "vm_pu"] - bfsw.res_bus.loc[common, "vm_pu"])
    return {
        "benchmark_source": "pandapower.networks.create_cigre_network_lv",
        "nr": _summary(nr),
        "bfsw": _summary(bfsw),
        "cross_algorithm": {
            "maximum_absolute_voltage_difference_pu": float(dv.max()),
            "mean_absolute_voltage_difference_pu": float(dv.mean()),
            "line_loss_difference_kw": abs(_summary(nr)["active_line_losses_kw"] - _summary(bfsw)["active_line_losses_kw"]),
        },
        "validation_note": "Cross-algorithm validation inside pandapower; an independently implemented solver remains required before claiming independent validation.",
    }


def write_results(path: str | Path) -> dict:
    result = compare_algorithms()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    net = build_cigre_lv()
    tables = destination.parents[1] / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in ("bus", "line", "trafo", "load", "switch", "ext_grid"):
        getattr(net, name).to_csv(tables / f"cigre_lv_{name}.csv", index_label="element_id")
    return result
