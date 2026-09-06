from pathlib import Path
from pprint import pprint
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.powerflow.opendss_validation import validate_cigre_with_opendss

if __name__ == "__main__":
    pprint(validate_cigre_with_opendss("results/simulations"))

