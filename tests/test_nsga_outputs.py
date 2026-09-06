from pathlib import Path
import pandas as pd


def test_nsga_experiment_artifacts():
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "optimization"
    front = pd.read_csv(out / "nsga_pareto_front.csv")
    assert front.seed.nunique() == 30
    assert set(["expected_cost", "active_losses", "voltage_deviation", "transformer_peak", "degradation"]).issubset(front.columns)
    assert len(front) > 0
    assert (out / "nsga_sensitivity.csv").exists()
    assert set(pd.read_csv(out / "scenario_convergence.csv").scenario_count) == {50, 100, 250, 500, 1000}
