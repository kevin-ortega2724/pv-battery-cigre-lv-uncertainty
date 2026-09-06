from pathlib import Path
from pprint import pprint
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.powerflow.timeseries import run_s0_timeseries
from src.visualization.diagnostics import powerflow_figures

if __name__ == "__main__":
    profiles = pd.read_csv("results/simulations/synthetic_dwellings_15min.csv", index_col=0, parse_dates=True)
    pprint(run_s0_timeseries(profiles, "results/simulations"))
    ts = pd.read_csv("results/simulations/cigre_s0_timeseries.csv", index_col=0, parse_dates=True)
    voltage = pd.read_csv("results/simulations/cigre_s0_bus_voltages.csv", index_col=0, parse_dates=True)
    powerflow_figures(ts, voltage, "paper/figures")

