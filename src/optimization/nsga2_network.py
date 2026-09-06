"""Network-aware NSGA-II evaluator for the S5/S6 experiment.

The optimizer keeps the CIGRE LV pandapower model as the evaluation substrate.
The chromosome contains one discrete bus index (rounded to the nearest of the
15 benchmark load buses), battery energy/power, PV rating, 96 quarter-hour
dispatch commands and 96 curtailment fractions.  Dispatch is state propagated
with the same 95% efficiencies and 10--90% SOC limits used by S2--S4.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandapower as pp
from pymoo.core.problem import ElementwiseProblem

from src.powerflow.cigre import build_cigre_lv
from src.powerflow.der_timeseries import synthetic_pv_profile
from src.powerflow.timeseries import normalized_load_multipliers
from src.optimization.scheduling import tariff


@dataclass
class ScenarioBank:
    profiles: pd.DataFrame
    seed: int = 20260904
    days: int = 100

    def __post_init__(self):
        net = build_cigre_lv()
        self.nominal_kw = net.load.p_mw.to_numpy() * 1000.0
        self.q_nominal = net.load.q_mvar.to_numpy()
        self.multipliers = normalized_load_multipliers(self.profiles, len(net.load))
        n = (len(self.multipliers) // 96) * 96
        self.multipliers = self.multipliers.iloc[:n]
        self.day_count = min(self.days, n // 96)
        self.index = self.multipliers.index
        self.pv = synthetic_pv_profile(self.index, self.seed).to_numpy()
        rng = np.random.default_rng(self.seed)
        self.day_pool = np.arange(self.day_count)
        self.fixed_days = rng.choice(self.day_pool, size=1000, replace=True)

    def sample(self, count: int, seed: int):
        rng = np.random.default_rng(seed)
        # Common-random-number bank: all policies receive these exact day IDs.
        return self.fixed_days[:count]

    def day(self, day_id: int, stride: int = 4):
        sl = slice(int(day_id) * 96, int(day_id + 1) * 96, stride)
        idx = self.index[sl]
        return (self.multipliers.iloc[sl].to_numpy(), self.pv[sl], idx)


def schmalstieg_aging_proxy(dispatch_kw: np.ndarray, soc: np.ndarray,
                            capacity_kwh: float, dt: float = .25) -> float:
    """Semi-empirical calendar/cycle proxy following Schmalstieg et al. [15].

    Equivalent full cycles and mean SOC are retained explicitly; the returned
    value is an aging index (fraction of rated life per experiment), suitable
    as a Pareto objective and not a replacement for cell-level identification.
    """
    if capacity_kwh <= 1e-9:
        return 0.0
    throughput = float(np.abs(dispatch_kw).sum() * dt)
    efc = throughput / (2.0 * capacity_kwh)
    mean_soc = float(np.mean(soc)) if len(soc) else .1
    cycle = 1.2e-4 * (max(efc, 0.0) ** 1.1) * (1.0 + 1.8 * abs(mean_soc - .5))
    calendar = 2.0e-5 * (len(soc) * dt / 24.0) * (0.7 + mean_soc)
    return float(cycle + calendar)


class NetworkAwareNSGAProblem(ElementwiseProblem):
    n_dispatch = 96
    n_curtail = 96

    def __init__(self, bank: ScenarioBank, scenario_days: np.ndarray,
                 pv_penetration: float = 1.0, forecast_error: float = 0.0,
                 duration_hours: float = 2.0, n_days: int = 4, stride: int = 4,
                 tariff_multiplier: float = 1.0):
        self.bank = bank
        self.tariff_multiplier = tariff_multiplier
        # FIX (2026-09-05): this evaluator previously truncated to 2 days at a
        # fixed stride=24 (one sample every 6 h) regardless of what was passed
        # in or documented -- the comment above claimed "four days" while the
        # code sliced [:2]. n_days/stride are now real constructor parameters:
        # n_days=4, stride=4 (hourly, i.e. every 4 of the 96 15-min steps) is
        # the tractable-but-documented compromise agreed with the author given
        # this environment's compute budget. It is still a reduced-fidelity
        # evaluation relative to the 100-day/15-min basis used by S0-S4, and
        # that must be stated explicitly wherever these results are reported.
        self.scenario_days = np.asarray(scenario_days, dtype=int)[:n_days]
        self.stride = int(stride)
        self.pv_penetration = pv_penetration
        self.forecast_error = forecast_error
        self.duration_hours = duration_hours
        self.base = build_cigre_lv()
        # FIX (2026-09-05): placement previously spanned all 15 CIGRE load
        # buses (residential + industrial + commercial). The optimizer then
        # regularly placed a large (up to 400 kW / 800 kWh) single DER behind
        # the 150 kVA industrial or 300 kVA commercial transformers -- which
        # this paper's PV/battery methodology (Sec. 4.1, Appendix C) never
        # covers -- causing >600% transformer overload regardless of sizing
        # and leaving 0% of the final population AC-feasible. Restricted to
        # the same six residential buses (R1, R11, R15, R16, R17, R18) that
        # S1-S4 use, per the sizing decision agreed with the author.
        self.load_buses = self.base.load.bus.to_numpy(dtype=int)[:6]
        self.pv_nominal = float(self.base.load.p_mw.iloc[:6].sum() * 1000.0 * .5)
        n_var = 4 + self.n_dispatch + self.n_curtail
        # placement, E(kWh), P(kW), PV(kW), dispatch[96], curtailment[96]
        super().__init__(n_var=n_var, n_obj=5, n_ieq_constr=4,
                         xl=np.r_[0., 0., 0., 0., np.full(192, -1.)],
                         xu=np.r_[5., 800., 400., self.pv_nominal * 2,
                                   np.ones(96), np.ones(96)])

    def _evaluate(self, x, out, *args, **kwargs):
        placement = int(np.clip(np.rint(x[0]), 0, 5))
        capacity = max(float(x[1]), 1e-6)
        power = max(float(x[2]), 0.0)
        pv_kw = max(float(x[3]), 0.0) * self.pv_penetration
        dispatch = np.asarray(x[4:100], dtype=float)
        curtail = np.clip(np.asarray(x[100:196], dtype=float), 0, 1)
        bus = int(self.load_buses[placement])
        totals = np.zeros(5); violation = np.zeros(4); n = 0
        soc_trace = []
        applied_dispatch_kw = []
        peak_max = 0.0
        # FIX (2026-09-05): dt is the REAL elapsed time between solved steps.
        # The previous code hardcoded 0.25 h (15 min) for both the SOC update
        # and the energy/cost accumulation regardless of stride, so at
        # stride=4 (hourly sampling) SOC drifted 4x too slowly and reported
        # energy/cost were understated by 4x. dt must scale with stride.
        dt = self.stride * 0.25
        for day in self.scenario_days:
            # Sampled points (every `stride`-th 15-min step) are solved by an
            # AC power flow; dispatch/curtailment chromosomes are indexed at
            # the matching stride so each solved step uses its own command.
            mult, pv_av, idx = self.bank.day(day, stride=self.stride)
            soc = .1 * capacity
            net = copy.deepcopy(self.base)
            sg = pp.create_sgen(net, bus=bus, p_mw=0., q_mvar=0., name="NSGA_PV_B")
            for j, timestamp in enumerate(idx):
                scale = mult[j].copy()
                if self.forecast_error:
                    scale *= np.exp(self.forecast_error * np.sin((j + int(day)) * .71))
                load_kw = self.bank.nominal_kw * scale
                net.load.loc[:, "p_mw"] = load_kw / 1000.
                net.load.loc[:, "q_mvar"] = self.bank.q_nominal * scale
                k = j * self.stride
                cmd = float(np.clip(dispatch[k], -1, 1)) * power
                if cmd >= 0:
                    actual = min(cmd, max((soc - .1 * capacity) * .95 / dt, 0.0))
                    soc -= actual * dt / .95
                    discharge, charge = actual, 0.
                else:
                    actual = min(-cmd, max((.9 * capacity - soc) / (.95 * dt), 0.0))
                    soc += actual * .95 * dt
                    discharge, charge = 0., actual
                pv = pv_kw * pv_av[j] * (1. - curtail[k])
                net.sgen.at[sg, "p_mw"] = (pv + discharge - charge) / 1000.
                try:
                    pp.runpp(net, algorithm="nr", init="flat" if n == 0 else "results",
                             tolerance_mva=1e-8, max_iteration=50,
                             calculate_voltage_angles=False, numba=False)
                    vm = net.res_bus.vm_pu.to_numpy()
                    losses = float(net.res_line.pl_mw.sum() * 1000.)
                    grid = float(net.res_ext_grid.p_mw.sum() * 1000.)
                    dev = float(np.abs(vm - 1.).mean())
                    peak = float(net.res_trafo.loading_percent.max())
                    # FIX (2026-09-05): dev and peak are per-step STATE values
                    # (a mean bus-voltage deviation, a worst-case transformer
                    # loading), not energy-like quantities. The previous code
                    # summed dev and peak across every solved step alongside
                    # cost/losses and then divided the total by the NUMBER OF
                    # DAYS -- with 24 solved steps/day (stride=4) that summed
                    # ~24 peak values and divided by ~4, inflating the
                    # reported "transformer_peak" objective roughly 6x (e.g.
                    # 630% instead of ~100%) regardless of the actual DER
                    # decision, and made the objective disagree with the
                    # separately (correctly) normalized constraint G[1]. dev
                    # is now averaged over all solved steps (n); peak is now
                    # the true running maximum, matching how S0-S4 report
                    # "maximum transformer loading".
                    totals[0] += tariff(pd.DatetimeIndex([timestamp]), self.tariff_multiplier)[0] * max(grid, 0.) * dt
                    totals[1] += losses * dt
                    totals[2] += dev
                    peak_max = max(peak_max, peak)
                    violation += [max(.95 - vm.min(), 0.) + max(vm.max() - 1.05, 0.),
                                  max(peak - 100., 0.) / 100., max(-grid, 0.) / 1000., 0.]
                    n += 1; soc_trace.append(soc / capacity)
                    applied_dispatch_kw.append(discharge - charge)
                except pp.LoadflowNotConverged:
                    violation[3] += 1.
        # FIX (2026-09-05): previously passed the raw 96-entry chromosome
        # (dispatch*power), most of which was never actually issued as a
        # command under the stride -- only every `stride`-th entry is applied
        # (see k = j*self.stride above). That inflated the throughput/aging
        # index by counting unused chromosome values. Use the dispatch power
        # actually applied at each solved step instead, at the matching dt.
        aging = schmalstieg_aging_proxy(np.asarray(applied_dispatch_kw), np.asarray(soc_trace), capacity, dt=dt)
        totals[4] = aging
        if n == 0:
            totals[:] = 1e6
        else:
            totals[0] /= len(self.scenario_days)
            totals[1] /= len(self.scenario_days)
            totals[2] /= n
            totals[3] = peak_max
        out["F"] = totals
        out["G"] = violation / max(n, 1)
        out["placement_bus"] = bus
        out["soc_min"] = float(min(soc_trace)) if soc_trace else 0.
        out["soc_max"] = float(max(soc_trace)) if soc_trace else 0.
