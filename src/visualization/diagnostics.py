from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecast_figures(predictions: pd.DataFrame, output: str | Path, horizon_label: str) -> dict:
    output = Path(output)
    qcols = [(0.1, "q10_kw"), (0.5, "q50_kw"), (0.9, "q90_kw")]
    observed = predictions["observed_kw"].to_numpy()
    empirical = [float(np.mean(observed <= predictions[col].to_numpy())) for _, col in qcols]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.plot([q for q, _ in qcols], empirical, marker="o", color="#0072B2")
    ax.set(xlabel="Nominal quantile", ylabel="Empirical frequency", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2)
    _save(fig, output / f"quantile_calibration_{horizon_label}")

    sample = predictions.iloc[: 7 * 96]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.fill_between(sample.index, sample["q10_kw"], sample["q90_kw"], color="#56B4E9", alpha=.3, label="80% interval")
    ax.plot(sample.index, sample["q50_kw"], color="#0072B2", linewidth=1, label="Median")
    ax.plot(sample.index, sample["observed_kw"], color="black", linewidth=.7, label="Observed")
    ax.set(ylabel="Active power (kW)")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=.15)
    _save(fig, output / f"probabilistic_forecast_{horizon_label}")
    return {f"q{int(q*100):02d}": value for (q, _), value in zip(qcols, empirical)}


def synthetic_profile_figure(profiles: pd.DataFrame, output: str | Path) -> None:
    sample = profiles.iloc[:96, : min(8, profiles.shape[1])]
    hours = np.arange(96) / 4
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for column in sample:
        ax.plot(hours, sample[column], linewidth=.8, alpha=.75)
    ax.set(xlabel="Hour", ylabel="Active power (kW)", xlim=(0, 23.75))
    ax.grid(alpha=.15)
    _save(fig, Path(output) / "synthetic_dwelling_profiles")


def powerflow_figures(timeseries: pd.DataFrame, voltages: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    sample = timeseries.iloc[: 7 * 96]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(sample.index, sample["minimum_voltage_pu"], color="#0072B2", linewidth=.9)
    ax.axhline(.95, color="#D55E00", linestyle="--", linewidth=1)
    ax.set(ylabel="Minimum voltage (p.u.)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(alpha=.15)
    _save(fig, output / "cigre_s0_minimum_voltage")

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(sample.index, sample["maximum_transformer_loading_percent"], color="#009E73", linewidth=.9)
    ax.axhline(100, color="#D55E00", linestyle="--", linewidth=1)
    ax.set(ylabel="Maximum transformer loading (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(alpha=.15)
    _save(fig, output / "cigre_s0_transformer_loading")

    worst = timeseries["minimum_voltage_pu"].idxmin()
    row = voltages.loc[worst]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(np.arange(len(row)), row, color="#56B4E9", width=.8)
    ax.axhline(.95, color="#D55E00", linestyle="--", linewidth=1)
    ax.set(xlabel="Bus index", ylabel="Voltage (p.u.)", ylim=(max(0.7, row.min()-.02), 1.01))
    ax.grid(axis="y", alpha=.15)
    _save(fig, output / "cigre_s0_worst_voltage_map")


def der_comparison_figure(summaries: list[dict], s2: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    labels = [item["scenario"].split()[0] for item in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fields = [("active_line_loss_energy_kwh", "Line losses (kWh)"),
              ("undervoltage_intervals", "Undervoltage intervals"),
              ("peak_grid_import_kw", "Peak import (kW)")]
    palette = ["#777777", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]
    colors = palette[:len(labels)]
    for ax, (field, ylabel) in zip(axes, fields):
        ax.bar(labels, [item[field] for item in summaries], color=colors)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.15)
    _save(fig, output / "cigre_scenario_comparison")

    sample = s2.iloc[: 7 * 96]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(sample.index, 100 * sample["mean_soc"], color="#009E73", linewidth=.9)
    ax.axhline(10, color="0.5", linestyle="--", linewidth=.8)
    ax.axhline(90, color="0.5", linestyle="--", linewidth=.8)
    ax.set(ylabel="Mean battery SOC (%)", ylim=(0, 100))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(alpha=.15)
    _save(fig, output / "cigre_s2_battery_soc")
