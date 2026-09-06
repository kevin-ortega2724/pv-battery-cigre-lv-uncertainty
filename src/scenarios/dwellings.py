from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _complete_days(series: pd.Series) -> pd.DataFrame:
    series = series.sort_index().dropna()
    groups = []
    for day, values in series.groupby(series.index.normalize()):
        if len(values) == 96 and values.index.minute.isin((0, 15, 30, 45)).all():
            groups.append(pd.Series(values.to_numpy(), name=day))
    if not groups:
        raise ValueError("No complete 96-interval days available")
    return pd.DataFrame(groups)


def generate_dwellings(
    measured: pd.Series,
    n_dwellings: int = 15,
    n_days: int = 100,
    seed: int = 20260904,
    amplitude_sigma: float = 0.25,
    max_shift_steps: int = 4,
    residual_fraction: float = 0.05,
    ar1_phi: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate labelled synthetic profiles by complete-day block bootstrap.

    A common source-day sequence preserves weather/season-like coincidence, while
    dwelling-specific neighbouring-day substitutions, amplitudes, time shifts and
    AR(1) residuals create diversity. Output is not independent measured data.
    """
    rng = np.random.default_rng(seed)
    days = _complete_days(measured)
    available = days.index.to_numpy()
    base_draws = rng.choice(available, size=n_days, replace=True)
    output = np.empty((n_days * 96, n_dwellings), dtype=float)
    metadata = []
    for dwelling in range(n_dwellings):
        scale = float(np.clip(rng.lognormal(mean=0.0, sigma=amplitude_sigma), 0.6, 1.6))
        shift = int(rng.integers(-max_shift_steps, max_shift_steps + 1))
        # Seventy percent of blocks share the common day; the remainder are
        # independently resampled to avoid implausibly identical households.
        use_common = rng.random(n_days) < 0.70
        chosen = np.where(use_common, base_draws, rng.choice(available, size=n_days, replace=True))
        profile = np.concatenate([days.loc[pd.Timestamp(day)].to_numpy() for day in chosen])
        profile = np.roll(profile.reshape(n_days, 96), shift, axis=1).reshape(-1) * scale
        innovation_sd = residual_fraction * max(float(np.mean(profile)), 1e-6)
        innovation = rng.normal(0.0, innovation_sd, size=len(profile))
        residual = np.empty_like(innovation)
        residual[0] = innovation[0]
        for t in range(1, len(residual)):
            residual[t] = ar1_phi * residual[t - 1] + innovation[t]
        output[:, dwelling] = np.clip(profile + residual, 0.0, None)
        metadata.append({"dwelling": dwelling + 1, "amplitude_k": scale,
                         "shift_steps_15min": shift, "ar1_phi": ar1_phi,
                         "innovation_sd_kw": innovation_sd})
    index = pd.date_range("2000-01-01", periods=n_days * 96, freq="15min")
    profiles = pd.DataFrame(output, index=index, columns=[f"dwelling_{i+1:02d}" for i in range(n_dwellings)])
    return profiles, pd.DataFrame(metadata)


def validation_metrics(profiles: pd.DataFrame) -> dict:
    daily = profiles.groupby(profiles.index.normalize())
    daily_energy = daily.sum() * 0.25
    daily_peak = daily.max()
    daily_mean = daily.mean()
    load_factor = daily_mean / daily_peak.replace(0, np.nan)
    ramps = profiles.diff().abs().iloc[1:]
    acf1 = profiles.apply(lambda x: x.autocorr(lag=1))
    corr = profiles.corr().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    aggregate_peak = profiles.sum(axis=1).max()
    sum_individual_peaks = profiles.max().sum()
    return {
        "label": "synthetic dwelling scenarios derived from measured data",
        "n_dwellings": int(profiles.shape[1]),
        "n_days": int(len(daily_energy)),
        "daily_energy_kwh": {"median": float(daily_energy.stack().median()), "iqr": float(daily_energy.stack().quantile(.75) - daily_energy.stack().quantile(.25))},
        "daily_peak_kw": {"median": float(daily_peak.stack().median()), "iqr": float(daily_peak.stack().quantile(.75) - daily_peak.stack().quantile(.25))},
        "load_factor": {"median": float(load_factor.stack().median()), "iqr": float(load_factor.stack().quantile(.75) - load_factor.stack().quantile(.25))},
        "absolute_ramp_kw": {"median": float(ramps.stack().median()), "p95": float(ramps.stack().quantile(.95))},
        "lag1_autocorrelation": {"median": float(acf1.median()), "range": [float(acf1.min()), float(acf1.max())]},
        "pairwise_correlation": {"median": float(np.median(upper)), "range": [float(np.min(upper)), float(np.max(upper))]},
        "peak_coincidence_factor": float(aggregate_peak / sum_individual_peaks),
    }


def compare_with_measured(profiles: pd.DataFrame, measured: pd.Series) -> dict:
    source = _complete_days(measured)
    source_energy = source.sum(axis=1) * 0.25
    source_peak = source.max(axis=1)
    source_factor = source.mean(axis=1) / source_peak.replace(0, np.nan)
    source_ramps = source.diff(axis=1).abs().iloc[:, 1:].stack()
    synthetic = validation_metrics(profiles)
    reference = {
        "daily_energy_kwh_median": float(source_energy.median()),
        "daily_peak_kw_median": float(source_peak.median()),
        "load_factor_median": float(source_factor.median()),
        "absolute_ramp_kw_p95": float(source_ramps.quantile(.95)),
    }
    ratios = {
        "daily_energy_median_ratio": synthetic["daily_energy_kwh"]["median"] / reference["daily_energy_kwh_median"],
        "daily_peak_median_ratio": synthetic["daily_peak_kw"]["median"] / reference["daily_peak_kw_median"],
        "load_factor_median_ratio": synthetic["load_factor"]["median"] / reference["load_factor_median"],
        "ramp_p95_ratio": synthetic["absolute_ramp_kw"]["p95"] / reference["absolute_ramp_kw_p95"],
    }
    return {"measured_single_dwelling_reference": reference, "synthetic_to_measured_ratios": ratios}


def write_scenarios(profiles, metadata, output_dir: str | Path, measured: pd.Series | None = None) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output / "synthetic_dwellings_15min.csv", index_label="synthetic_time")
    metadata.to_csv(output / "synthetic_dwellings_parameters.csv", index=False)
    metrics = validation_metrics(profiles)
    if measured is not None:
        metrics["source_comparison"] = compare_with_measured(profiles, measured)
    (output / "synthetic_dwellings_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
