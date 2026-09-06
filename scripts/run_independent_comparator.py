"""Genuinely independent weighted-sum comparator for the NSGA-II study.

FIX (2026-09-05): the previous weighted_sum_comparator.csv reselected a
candidate from the SAME NSGA-II population (front.sort_values(...).groupby
('seed').first()) -- not an independent optimizer, as ESTADO_ENTREGA.md
documents.

NOTE ON METHOD: the chromosome has 196 dimensions (placement, capacity,
power, PV rating, plus 96 dispatch + 96 curtailment values). scipy's
differential_evolution scales its population with dimensionality
(popsize * len(x) individuals per generation), which would need ~1,500+
AC-power-flow evaluations PER GENERATION at this dimensionality -- not
tractable in this environment. Instead this script uses independent Latin
hypercube random search: N candidates are drawn (no relation to NSGA-II's
population, crossover, or mutation), each evaluated once on the SAME
scalarized objective, and the best-by-score candidate is kept per seed. This
is a weaker search than a tuned metaheuristic, but it is a real, independently
implemented baseline rather than a reselection from NSGA-II's own results,
and the evaluation budget (N=150/seed) is disclosed plainly.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import numpy as np, pandas as pd
from scipy.stats import qmc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization.nsga2_network import ScenarioBank, NetworkAwareNSGAProblem

N_SAMPLES = 150


def evaluate(x, prob, scale):
    out = {}
    prob._evaluate(np.asarray(x, dtype=float), out)
    f = np.asarray(out["F"], dtype=float)
    g = np.asarray(out["G"], dtype=float)
    penalty = 1e3 * float(np.maximum(g, 0).sum())
    return float((f / scale).sum() + penalty), f, g


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "optimization"
    front = pd.read_csv(out_dir / "nsga_pareto_front.csv")
    cols = ["expected_cost", "active_losses", "voltage_deviation", "transformer_peak", "degradation"]
    scale = front[cols].replace(0, np.nan).abs().median().to_numpy()
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    profiles = pd.read_csv(root / "results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
    bank = ScenarioBank(profiles, days=100)
    seeds = sorted(front["seed"].unique().tolist())
    rows = []
    for seed in seeds:
        days = bank.sample(50, int(seed))
        prob = NetworkAwareNSGAProblem(bank, days, pv_penetration=1.0, forecast_error=0.0, duration_hours=2.0)
        sampler = qmc.LatinHypercube(d=len(prob.xl), seed=int(seed))
        u = sampler.random(N_SAMPLES)
        candidates = qmc.scale(u, prob.xl, prob.xu)
        t0 = time.time()
        best_score, best_f, best_g, best_x = np.inf, None, None, None
        for x in candidates:
            score, f, g = evaluate(x, prob, scale)
            if score < best_score:
                best_score, best_f, best_g, best_x = score, f, g, x
        elapsed = time.time() - t0
        rows.append({
            "seed": int(seed), "optimizer": f"independent_random_search (LHS, N={N_SAMPLES})",
            "placement_index": int(np.rint(best_x[0])), "energy_kwh": float(best_x[1]),
            "power_kw": float(best_x[2]), "pv_kw": float(best_x[3]),
            "expected_cost": float(best_f[0]), "active_losses": float(best_f[1]),
            "voltage_deviation": float(best_f[2]), "transformer_peak": float(best_f[3]),
            "degradation": float(best_f[4]), "constraint_violation": float(np.maximum(best_g, 0).sum()),
            # near-feasible threshold, see fix_nsga_statistics.py note
            "feasible": bool(np.maximum(best_g, 0).sum() <= 0.001),
            "weighted_sum": float(best_score), "n_evaluated": N_SAMPLES, "elapsed_s": float(elapsed),
        })
        print(f"seed {seed}: best_score={best_score:.3f} feasible={rows[-1]['feasible']} elapsed={elapsed:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "weighted_sum_comparator.csv", index=False)


if __name__ == "__main__":
    main()
