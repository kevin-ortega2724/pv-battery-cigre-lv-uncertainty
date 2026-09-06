import argparse
from pathlib import Path
from pprint import pprint
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.pilot import run_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/household_15min.csv")
    parser.add_argument("--horizon-steps", type=int, default=4)
    args = parser.parse_args()
    pprint(run_pilot(args.processed, "results/metrics", args.horizon_steps))

