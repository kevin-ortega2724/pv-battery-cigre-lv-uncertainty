"""Real tariff sensitivity: re-optimizes NSGA-II under each tariff multiplier.

FIX (2026-09-05): the previous tariff_sensitivity.csv only rescaled the
ALREADY-OPTIMIZED baseline cost by the multiplier at fixed dispatch -- the
policy itself never responded to the tariff change, so it could not show
whether a different price signal shifts the optimal placement/dispatch.
This script re-runs NSGA2 with tariff_multiplier in {0.75, 1.0, 1.25, 1.5},
letting the optimizer actually re-decide dispatch under each tariff. Scope is
deliberately small (2 seeds, population 8, 10 generations -- lighter than the
main run) so this auxiliary sweep stays a modest addition to the session's
compute budget rather than tripling it; the multiplier=1.0 case reuses the
completed main run's seeds 1-2 rather than re-solving it. This trades some
precision for tractability -- treat the sensitivity DIRECTION as the main
finding, not the exact magnitudes.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import numpy as np, pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization.nsga2_network import ScenarioBank, NetworkAwareNSGAProblem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "optimization"
SEEDS = (1, 2)
POP, GEN = 8, 10


def run_one(bank, seed, multiplier):
    days = bank.sample(50, seed)
    prob = NetworkAwareNSGAProblem(bank, days, pv_penetration=1.0, forecast_error=0.0,
                                    duration_hours=2.0, tariff_multiplier=multiplier)
    alg = NSGA2(pop_size=POP, sampling=FloatRandomSampling(), crossover=SBX(prob=.9, eta=15), mutation=PM(eta=20))
    res = minimize(prob, alg, ("n_gen", GEN), seed=int(seed), verbose=False)
    if res.X is None:
        X, F, G = np.atleast_2d(res.pop.get("X")), np.atleast_2d(res.pop.get("F")), np.atleast_2d(res.pop.get("G"))
    else:
        X, F, G = np.atleast_2d(res.X), np.atleast_2d(res.F), np.atleast_2d(res.G)
    rows = []
    for i in range(len(F)):
        feas = bool(np.maximum(G[i], 0).sum() <= 1e-9)
        rows.append({"seed": seed, "tariff_multiplier": multiplier,
                     "expected_cost": float(F[i, 0]), "active_losses": float(F[i, 1]),
                     "voltage_deviation": float(F[i, 2]), "transformer_peak": float(F[i, 3]),
                     "degradation": float(F[i, 4]), "feasible": feas})
    return rows


def main():
    profiles = pd.read_csv(ROOT / "results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
    bank = ScenarioBank(profiles, days=100)

    baseline = pd.read_csv(OUT / "nsga_pareto_front.csv")
    baseline = baseline[baseline["seed"].isin(SEEDS) & baseline["feasible"]]
    rows = [{"seed": int(r.seed), "tariff_multiplier": 1.0, "expected_cost": r.expected_cost,
             "active_losses": r.active_losses, "voltage_deviation": r.voltage_deviation,
             "transformer_peak": r.transformer_peak, "degradation": r.degradation, "feasible": True}
            for r in baseline.itertuples()]

    for mult in (0.75, 1.25, 1.5):
        for seed in SEEDS:
            t0 = time.time()
            rows += run_one(bank, seed, mult)
            print(f"multiplier={mult} seed={seed} done in {time.time()-t0:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "tariff_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
