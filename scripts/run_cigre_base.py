from pathlib import Path
from pprint import pprint
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.powerflow.cigre import write_results

if __name__ == "__main__":
    pprint(write_results("results/simulations/cigre_base_powerflow.json"))
