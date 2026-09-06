from pathlib import Path
from pprint import pprint
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.powerflow.forecast_control import run_forecast_control

root = Path("results/simulations")
profiles = pd.read_csv(root / "synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
for scenario in ("S3", "S4"):
    schedule = pd.read_csv(root / f"{scenario.lower()}_battery_schedule.csv", index_col=0, parse_dates=True)
    pprint(run_forecast_control(profiles, schedule, root, scenario))

