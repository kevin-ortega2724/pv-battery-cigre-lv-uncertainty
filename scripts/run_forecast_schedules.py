from pathlib import Path
from pprint import pprint
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.optimization.scheduling import make_forecast_schedules

profiles = pd.read_csv("results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
pprint(make_forecast_schedules(profiles, "results/simulations"))

