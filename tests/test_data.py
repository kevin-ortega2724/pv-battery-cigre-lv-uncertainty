import pandas as pd

from src.data.household import aggregate_15min, quality_report


def _frame(missing_last=False):
    ts = pd.date_range("2010-01-01", periods=15, freq="min")
    data = {"timestamp": ts, "Global_active_power": 1.0, "Global_reactive_power": 0.1,
            "Voltage": 230.0, "Global_intensity": 4.5, "Sub_metering_1": 2.0,
            "Sub_metering_2": 3.0, "Sub_metering_3": 5.0}
    frame = pd.DataFrame(data)
    if missing_last:
        frame.loc[14, "Global_active_power"] = float("nan")
    return frame


def test_interval_with_14_measurements_is_retained():
    assert len(aggregate_15min(_frame(True))) == 1


def test_interval_with_13_measurements_is_excluded():
    frame = _frame(True)
    frame.loc[13, "Global_active_power"] = float("nan")
    assert aggregate_15min(frame).empty


def test_residual_units():
    row = aggregate_15min(_frame()).iloc[0]
    assert abs(row["Residual_active_power_kW"] - 0.4) < 1e-12
    assert abs(row["Residual_energy_Wh"] - 100.0) < 1e-12


def test_quality_reports_duplicates_and_gap():
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True).sort_values("timestamp")
    report = quality_report(frame)
    assert report["duplicate_timestamps"] == 1


def test_quality_counts_consecutive_missing_measurements_as_outage():
    frame = _frame()
    frame.loc[4:6, "Global_active_power"] = float("nan")
    report = quality_report(frame)
    assert report["numeric_missing_runs"] == 1
    assert report["largest_numeric_outage_minutes"] == 3
