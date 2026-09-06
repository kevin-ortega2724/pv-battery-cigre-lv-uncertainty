import pandas as pd


def test_saved_schedules_are_time_aligned_and_non_simultaneous():
    for scenario in ("s3", "s4"):
        path = f"results/simulations/{scenario}_battery_schedule.csv"
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        assert len(frame) == 9600
        assert not ((frame.charge_kw > 1e-8) & (frame.discharge_kw > 1e-8)).any()
