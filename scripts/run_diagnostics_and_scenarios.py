from pathlib import Path
from pprint import pprint
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.scenarios.dwellings import generate_dwellings, write_scenarios
from src.visualization.diagnostics import forecast_figures, synthetic_profile_figure

if __name__ == "__main__":
    figures = Path("paper/figures")
    for steps, label in ((1, "15min"), (4, "1h"), (24, "6h"), (96, "24h")):
        predictions = pd.read_csv(f"results/metrics/pilot_predictions_h{steps}.csv", index_col=0, parse_dates=True)
        print(label, forecast_figures(predictions, figures, label))
    measured = pd.read_csv("data/processed/household_15min.csv", index_col=0, parse_dates=True)["Global_active_power"]
    profiles, metadata = generate_dwellings(measured)
    pprint(write_scenarios(profiles, metadata, "results/simulations", measured))
    synthetic_profile_figure(profiles, figures)
