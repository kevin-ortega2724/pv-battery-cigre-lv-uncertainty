from pathlib import Path
from pprint import pprint
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.powerflow.der_timeseries import run_der_scenario

profiles = pd.read_csv("results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
scenarios = tuple(sys.argv[1:]) or ("S1", "S2")
for scenario in scenarios:
    pprint(run_der_scenario(profiles, "results/simulations", scenario))
