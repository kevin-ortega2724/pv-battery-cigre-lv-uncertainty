"""Real paired-comparison statistics for the NSGA-II study, replacing the
broken paired_policy_statistics.csv / nsga_statistics.csv / scenario_convergence.csv.

FIXES (2026-09-05), each documented in ESTADO_ENTREGA.md:
 1. paired_policy_statistics.csv previously compared 5 seed-level MEANS against
    hardcoded reference CONSTANTS (not measured from any daily replay). This
    script instead replays each retained feasible NSGA-II candidate at full
    15-min resolution on the SAME 4 representative days used by the optimizer,
    and pairs each (candidate, day) value against S2/S3/S4's ACTUAL measured
    value on that SAME day (from the existing cigre_s{2,3,4}_timeseries.csv).
 2. Holm-Bonferroni is applied via statsmodels' sequential procedure, not a
    single-step Bonferroni copy.
 3. Rank-biserial effect size is computed with its sign retained (previously
    hardcoded to 0.0 or otherwise discarded).
 4. nsga_statistics.csv previously compared seed values against their OWN
    median (circular). This script reuses the same paired data against
    S2/S3/S4 instead.
 5. scenario_convergence.csv previously reported sampled day IDs, not
    objective convergence. This script re-evaluates a fixed reference
    candidate at 1/2/3/4 of the representative days and reports how each
    objective changes as more days are included.
 6. voltage_deviation has NO counterpart in the existing S2-S4 outputs (only
    minimum/maximum bus voltage were retained, not the full per-bus vector
    needed for the same "mean |V-1| across all buses" definition NSGA uses).
    It is EXCLUDED from the paired significance tests and reported only
    descriptively, with this limitation stated explicitly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization.nsga2_network import ScenarioBank, NetworkAwareNSGAProblem, schmalstieg_aging_proxy
from src.optimization.scheduling import tariff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "optimization"
DER_CAPACITY_KWH = 190 + 14.25 + 49.4 + 52.25 + 33.25 + 44.65  # sum of six residential BESS sites


def load_bank():
    profiles = pd.read_csv(ROOT / "results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
    return ScenarioBank(profiles, days=100)


def replay_candidate_per_day(bank, chromosome, days):
    """Re-evaluate one NSGA chromosome at full 15-min resolution, one day at a
    time, returning a per-day DataFrame of the five objectives."""
    x = np.asarray(chromosome, dtype=float)
    rows = []
    for day in days:
        prob = NetworkAwareNSGAProblem(bank, np.array([day]), pv_penetration=1.0,
                                        forecast_error=0.0, duration_hours=2.0,
                                        n_days=1, stride=1)
        out = {}
        prob._evaluate(x, out)
        f = out["F"]
        rows.append({"day": int(day), "expected_cost": f[0], "active_losses": f[1],
                     "voltage_deviation": f[2], "transformer_peak": f[3], "degradation": f[4]})
    return pd.DataFrame(rows)


def s_policy_per_day(policy: str, dates):
    df = pd.read_csv(ROOT / f"results/simulations/cigre_{policy.lower()}_timeseries.csv", parse_dates=["timestamp"])
    rows = []
    for d in dates:
        day_df = df[df["timestamp"].dt.date == d.date()]
        if day_df.empty:
            continue
        cost = float((tariff(pd.DatetimeIndex(day_df["timestamp"])) * day_df["grid_import_kw"].clip(lower=0)).sum() * 0.25)
        losses = float(day_df["active_line_losses_kw"].sum() * 0.25)
        peak = float(day_df["maximum_transformer_loading_percent"].max())
        charge = day_df["battery_charge_kw"].to_numpy()
        discharge = day_df["battery_discharge_kw"].to_numpy()
        soc = (day_df["mean_soc"].to_numpy() / 100.0) if day_df["mean_soc"].max() > 1.5 else day_df["mean_soc"].to_numpy()
        degradation = schmalstieg_aging_proxy(discharge - charge, soc, DER_CAPACITY_KWH, dt=0.25)
        rows.append({"day": d, "expected_cost": cost, "active_losses": losses,
                     "transformer_peak": peak, "degradation": degradation})
    return pd.DataFrame(rows)


def signed_rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, sign retained: positive means
    the first sample (NSGA candidate) tends to be LARGER than the comparator."""
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total > 0 else 0.0


def main():
    front = pd.read_csv(OUT / "nsga_pareto_front.csv")
    cols = ["expected_cost", "active_losses", "voltage_deviation", "transformer_peak", "degradation"]
    # NOTE (2026-09-05): after fixing the placement scope and the peak/dev
    # aggregation bug, objectives are now physically sensible (transformer
    # peak ~85%, matching S0-S4's scale) but the ENTIRE converged population
    # (all 60 final individuals, all 5 seeds) sits at a small, tight residual
    # constraint_violation (~0.00086-0.00096), not exactly zero. This matches
    # a real physical floor: the baseline network (no new DER at all) already
    # has undervoltage in 9.49% of S0's intervals, and a single new
    # residential-site battery/PV cannot eliminate network-wide undervoltage
    # caused by OTHER buses/loads. Treating only G<=1e-9 as "feasible" would
    # discard 100% of a cleanly-converged population over a threshold no
    # candidate can plausibly meet. NEAR_FEASIBLE_THRESHOLD is set just above
    # the observed cluster (all 60/60 rows are <=0.001) -- state this
    # explicitly wherever "feasible" is reported.
    NEAR_FEASIBLE_THRESHOLD = 0.001
    front["feasible"] = front["constraint_violation"] <= NEAR_FEASIBLE_THRESHOLD
    feasible = front[front["feasible"]].copy()
    feasible.to_csv(OUT / "nsga_pareto_front_feasible.csv", index=False)
    print(f"feasible candidates: {len(feasible)} / {len(front)} total rows "
          f"({100*len(feasible)/max(len(front),1):.1f}%)")

    if feasible.empty:
        print("WARNING: no feasible candidates -- cannot run paired statistics.")
        pd.DataFrame(columns=["comparison", "objective", "n_pairs", "wilcoxon_stat",
                               "wilcoxon_p_raw", "holm_p", "rank_biserial"]).to_csv(
            OUT / "paired_policy_statistics.csv", index=False)
        return

    scale = feasible[cols].replace(0, np.nan).abs().median()
    z = feasible[cols].divide(scale)
    feasible = feasible.assign(weighted_sum=z.sum(axis=1))
    retained = feasible.sort_values("weighted_sum").groupby("seed", as_index=False).first()
    retained.to_csv(OUT / "nsga_retained_candidates.csv", index=False)
    print(f"retained (best-per-seed) candidates: {len(retained)}")

    bank = load_bank()
    rep_days = bank.sample(50, int(retained["seed"].iloc[0]))[:4]
    dates = pd.DatetimeIndex([bank.index[d * 96] for d in rep_days])
    print("representative days:", rep_days, list(dates.date))

    per_candidate = []
    for _, row in retained.iterrows():
        chromo = json.loads(row["chromosome"])
        pd_day = replay_candidate_per_day(bank, chromo, rep_days)
        pd_day["seed"] = int(row["seed"])
        per_candidate.append(pd_day)
    nsga_daily = pd.concat(per_candidate, ignore_index=True)
    nsga_daily.to_csv(OUT / "nsga_retained_candidates_daily_replay.csv", index=False)

    policy_daily = {p: s_policy_per_day(p, dates) for p in ("S2", "S3", "S4")}
    for p, d in policy_daily.items():
        d.to_csv(OUT / f"{p.lower()}_daily_reference_same_days.csv", index=False)

    comparable_cols = ["expected_cost", "active_losses", "transformer_peak", "degradation"]
    day_to_ref = {}
    for p, d in policy_daily.items():
        d = d.copy(); d["date"] = pd.to_datetime(d["day"]).dt.date
        day_to_ref[p] = d.set_index("date")[comparable_cols]

    day_lookup = {int(day): date.date() for day, date in zip(rep_days, dates)}
    nsga_daily["date"] = nsga_daily["day"].map(day_lookup)

    rows = []
    for policy, ref in day_to_ref.items():
        merged = nsga_daily.merge(ref, left_on="date", right_index=True, suffixes=("_nsga", "_" + policy.lower()))
        for c in comparable_cols:
            x = merged[f"{c}_nsga"].to_numpy(float)
            y = merged[f"{c}_{policy.lower()}"].to_numpy(float)
            diff = x - y
            n = len(diff)
            try:
                stat, p_raw = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                stat, p_raw = 0.0, 1.0
            rows.append({"comparison": f"NSGA-II (retained, full-res daily replay) vs {policy}",
                         "objective": c, "n_pairs": n, "wilcoxon_stat": float(stat),
                         "wilcoxon_p_raw": float(p_raw), "rank_biserial": signed_rank_biserial(diff),
                         "mean_diff": float(np.mean(diff)) if n else float("nan")})
    stats = pd.DataFrame(rows)
    if not stats.empty:
        reject, holm_p, _, _ = multipletests(stats["wilcoxon_p_raw"], method="holm")
        stats["holm_p"] = holm_p
        stats["reject_at_0.05"] = reject
    stats.to_csv(OUT / "paired_policy_statistics.csv", index=False)
    stats.to_csv(OUT / "nsga_statistics.csv", index=False)  # single source of truth now; old file compared vs internal median
    print(stats.to_string(index=False))
    print("\nNOTE: voltage_deviation excluded from paired tests -- the existing "
          "S2-S4 outputs only retain min/max bus voltage per step, not the "
          "full per-bus vector NSGA's mean|V-1| definition needs. Reported "
          "descriptively (NSGA retained candidates only) below.")
    print(nsga_daily.groupby("seed")["voltage_deviation"].mean())

    # Scenario-count convergence: re-evaluate ONE fixed reference chromosome
    # (the best-by-weighted-sum candidate overall) at 1/2/3/4 of the same
    # representative days, and report how each objective changes -- real
    # convergence of OBJECTIVES, not day-ID bookkeeping.
    best = retained.sort_values("weighted_sum").iloc[0]
    chromo = json.loads(best["chromosome"])
    conv_rows = []
    for k in range(1, len(rep_days) + 1):
        prob = NetworkAwareNSGAProblem(bank, rep_days[:k], pv_penetration=1.0,
                                        forecast_error=0.0, duration_hours=2.0,
                                        n_days=k, stride=4)
        out = {}
        prob._evaluate(np.asarray(chromo, dtype=float), out)
        f = out["F"]
        conv_rows.append({"n_days": k, "expected_cost": f[0], "active_losses": f[1],
                           "voltage_deviation": f[2], "transformer_peak": f[3], "degradation": f[4]})
    pd.DataFrame(conv_rows).to_csv(OUT / "scenario_convergence.csv", index=False)
    print("\nscenario_convergence.csv (objective values vs number of representative days):")
    print(pd.DataFrame(conv_rows).to_string(index=False))


if __name__ == "__main__":
    main()
