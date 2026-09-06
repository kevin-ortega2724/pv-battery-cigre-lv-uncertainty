from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the UCI household file without modifying it."""
    frame = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    frame["timestamp"] = pd.to_datetime(
        frame["Date"] + " " + frame["Time"], format="%d/%m/%Y %H:%M:%S", errors="raise"
    )
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def quality_report(frame: pd.DataFrame) -> dict:
    ts = frame["timestamp"]
    gaps = ts.diff().dropna()
    missing_any = frame[NUMERIC_COLUMNS].isna().any(axis=1)
    # The UCI file preserves minute timestamps during many outages and stores
    # question marks in every numeric column. These must not be mistaken for
    # complete temporal coverage merely because timestamp differences are 1 min.
    run_id = missing_any.ne(missing_any.shift(fill_value=False)).cumsum()
    missing_runs = missing_any.groupby(run_id).agg(["first", "size"])
    missing_run_lengths = missing_runs.loc[missing_runs["first"], "size"]
    suspicious = {
        "negative_active_power_rows": int((frame["Global_active_power"] < 0).sum()),
        "nonpositive_voltage_rows": int((frame["Voltage"] <= 0).sum()),
        "negative_submeter_rows": int((frame[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]] < 0).any(axis=1).sum()),
    }
    return {
        "rows": int(len(frame)),
        "start": ts.min().isoformat(),
        "end": ts.max().isoformat(),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "rows_with_any_numeric_missing": int(missing_any.sum()),
        "missing_by_column": {k: int(v) for k, v in frame[NUMERIC_COLUMNS].isna().sum().items()},
        "gaps_over_one_minute": int((gaps > pd.Timedelta(minutes=1)).sum()),
        "largest_gap_minutes": float(gaps.max() / pd.Timedelta(minutes=1)),
        "numeric_missing_runs": int(len(missing_run_lengths)),
        "largest_numeric_outage_minutes": int(missing_run_lengths.max()) if len(missing_run_lengths) else 0,
        "largest_numeric_outage_days": float(missing_run_lengths.max() / 1440.0) if len(missing_run_lengths) else 0.0,
        "suspicious": suspicious,
    }


def aggregate_15min(frame: pd.DataFrame, minimum_count: int = 14) -> pd.DataFrame:
    """Aggregate power by mean; retain bins with >=14 observed active-power minutes."""
    indexed = frame.set_index("timestamp")
    counts = indexed["Global_active_power"].resample("15min").count()
    means = indexed[NUMERIC_COLUMNS].resample("15min").mean()
    means["observed_minutes"] = counts
    valid = means.loc[counts >= minimum_count].copy()
    # Sub-meter readings are Wh over one minute; multiply by 60 to obtain W.
    valid["Residual_active_power_kW"] = (
        valid["Global_active_power"]
        - 60.0 * (valid["Sub_metering_1"] + valid["Sub_metering_2"] + valid["Sub_metering_3"]) / 1000.0
    )
    valid["Residual_energy_Wh"] = valid["Residual_active_power_kW"] * 250.0
    return valid


def write_audit(frame: pd.DataFrame, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = quality_report(frame)
    (output / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate_15min(frame).describe(include="all").to_csv(output / "descriptive_statistics.csv")
    return report
