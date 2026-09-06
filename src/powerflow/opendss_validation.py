from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandapower as pp

from .cigre import build_cigre_lv


def _bus(index: int) -> str:
    return f"bus_{index}"


def build_dss_commands(net) -> list[str]:
    commands = [
        "Clear",
        "Set DefaultBaseFrequency=50",
        "New Circuit.CIGRE_LV bus1=bus_0 phases=3 basekv=20 pu=1 angle=0 frequency=50 mvasc3=1e9 mvasc1=1e9 x1r1=1 x0r0=1",
    ]
    for idx, switch in net.switch.loc[net.switch.closed & (net.switch.et == "b")].iterrows():
        commands.append(
            f"New Line.Switch_{idx} phases=3 bus1={_bus(int(switch.bus))} bus2={_bus(int(switch.element))} "
            "r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6 c1=0 c0=0 length=0.001 units=km"
        )
    for idx, line in net.line.loc[net.line.in_service].iterrows():
        commands.append(
            f"New Line.L{idx} phases=3 bus1={_bus(int(line.from_bus))} bus2={_bus(int(line.to_bus))} "
            f"r1={line.r_ohm_per_km:.12g} x1={line.x_ohm_per_km:.12g} "
            f"r0={line.r_ohm_per_km:.12g} x0={line.x_ohm_per_km:.12g} "
            f"c1={line.c_nf_per_km:.12g} c0={line.c_nf_per_km:.12g} "
            f"length={line.length_km:.12g} units=km"
        )
    for idx, trafo in net.trafo.loc[net.trafo.in_service].iterrows():
        xhl = math.sqrt(max(trafo.vk_percent**2 - trafo.vkr_percent**2, 0.0))
        half_r = trafo.vkr_percent / 2
        commands.append(
            f"New Transformer.T{idx} phases=3 windings=2 xhl={xhl:.12g} %Rs=[{half_r:.12g},{half_r:.12g}] "
            f"wdg=1 bus={_bus(int(trafo.hv_bus))} conn=delta kv={trafo.vn_hv_kv:.12g} kva={trafo.sn_mva*1000:.12g} "
            f"wdg=2 bus={_bus(int(trafo.lv_bus))} conn=wye kv={trafo.vn_lv_kv:.12g} kva={trafo.sn_mva*1000:.12g}"
        )
    for idx, load in net.load.loc[net.load.in_service].iterrows():
        kv = net.bus.at[int(load.bus), "vn_kv"]
        commands.append(
            f"New Load.Load_{idx} phases=3 bus1={_bus(int(load.bus))} conn=wye model=1 "
            f"kv={kv:.12g} kw={load.p_mw*1000:.12g} kvar={load.q_mvar*1000:.12g} vminpu=0 vmaxpu=2"
        )
    commands += ["Set voltagebases=[20, 0.4]", "CalcVoltageBases", "Set mode=snapshot", "Solve"]
    return commands


def _dss_bus_voltage(dss, bus_name: str) -> float:
    dss.Circuit.SetActiveBus(bus_name)
    values = dss.Bus.puVmagAngle()
    magnitudes = np.asarray(values[0::2], dtype=float)
    return float(magnitudes.mean())


def validate_cigre_with_opendss(output_dir: str | Path) -> dict:
    import opendssdirect as dss
    net = build_cigre_lv()
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-10, max_iteration=100,
             calculate_voltage_angles=True, numba=False)
    commands = build_dss_commands(net)
    for command in commands:
        dss.Text.Command(command)
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge")
    dss_vm = np.array([_dss_bus_voltage(dss, _bus(i)) for i in net.bus.index])
    pp_vm = net.res_bus.loc[net.bus.index, "vm_pu"].to_numpy()
    difference = np.abs(pp_vm - dss_vm)
    dss_losses_kw = dss.Circuit.Losses()[0] / 1000
    pp_losses_kw = (net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * 1000
    result = {
        "pandapower_version": pp.__version__,
        "opendssdirect_version": getattr(dss, "__version__", "0.9.4"),
        "both_converged": bool(net.converged and dss.Solution.Converged()),
        "maximum_absolute_voltage_difference_pu": float(difference.max()),
        "mean_absolute_voltage_difference_pu": float(difference.mean()),
        "worst_bus": int(net.bus.index[difference.argmax()]),
        "pandapower_minimum_voltage_pu": float(pp_vm.min()),
        "opendss_minimum_voltage_pu": float(dss_vm.min()),
        "pandapower_total_losses_kw": float(pp_losses_kw),
        "opendss_total_losses_kw": float(dss_losses_kw),
        "absolute_loss_difference_kw": float(abs(pp_losses_kw - dss_losses_kw)),
        "relative_loss_difference_percent": float(100 * abs(pp_losses_kw - dss_losses_kw) / pp_losses_kw),
        "translation_assumptions": [
            "Balanced three-phase positive-sequence comparison.",
            "CIGRE bus-bus switches represented by negligible-impedance closed lines.",
            "Transformer total vkr split equally between windings; xhl derived from vk and vkr.",
            "OpenDSS load voltage fallback disabled (vminpu=0, vmaxpu=2) to match pandapower constant-PQ loads.",
            "OpenDSS source made effectively ideal because pandapower ext_grid short-circuit fields are ignored in load-flow calculations.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cigre_lv_opendss.dss").write_text("\n".join(commands), encoding="utf-8")
    (output / "cigre_opendss_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
