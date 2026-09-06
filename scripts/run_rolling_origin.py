from pathlib import Path
from pprint import pprint
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting.rolling import rolling_origin_2009

if __name__ == "__main__":
    pprint(rolling_origin_2009("data/processed/household_15min.csv", "results/metrics/rolling_origin_1h.json"))
