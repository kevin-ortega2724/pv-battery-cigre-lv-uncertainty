import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.powerflow.timeseries import summarize_timeseries

path = Path("results/simulations/cigre_s0_timeseries.csv")
frame = pd.read_csv(path, index_col=0, parse_dates=True)
summary = summarize_timeseries(frame)
Path("results/simulations/cigre_s0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
