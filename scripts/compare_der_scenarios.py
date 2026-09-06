import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.visualization.diagnostics import der_comparison_figure

root = Path("results/simulations")
scenario_ids = ("s0", "s1", "s2", "s3", "s4")
summaries = [json.loads((root / f"cigre_{s}_summary.json").read_text(encoding="utf-8")) for s in scenario_ids]
for item in summaries:
    item.setdefault("undervoltage_intervals", item.get("intervals_with_undervoltage"))
s2 = pd.read_csv(root / "cigre_s2_timeseries.csv", index_col=0, parse_dates=True)
checks = {}
for scenario in ("s2", "s3", "s4"):
    frame = pd.read_csv(root / f"cigre_{scenario}_timeseries.csv", index_col=0, parse_dates=True)
    minimum_soc = frame["minimum_soc"] if "minimum_soc" in frame else frame["mean_soc"]
    maximum_soc = frame["maximum_soc"] if "maximum_soc" in frame else frame["mean_soc"]
    checks[scenario] = {
        "minimum_soc": float(minimum_soc.min()),
        "maximum_soc": float(maximum_soc.max()),
        "simultaneous_charge_discharge_intervals": int(((frame.battery_charge_kw > 1e-9) & (frame.battery_discharge_kw > 1e-9)).sum()),
        "converged_intervals": int(frame.converged.sum()),
    }
baseline = summaries[0]
comparison = []
for item in summaries:
    comparison.append({
        "scenario": item["scenario"],
        "loss_reduction_vs_s0_percent": 100 * (baseline["active_line_loss_energy_kwh"] - item["active_line_loss_energy_kwh"]) / baseline["active_line_loss_energy_kwh"],
        "undervoltage_reduction_vs_s0_percent": 100 * (baseline["intervals_with_undervoltage"] - item.get("undervoltage_intervals", item.get("intervals_with_undervoltage"))) / baseline["intervals_with_undervoltage"],
        "peak_reduction_vs_s0_percent": 100 * (baseline["peak_grid_import_kw"] - item["peak_grid_import_kw"]) / baseline["peak_grid_import_kw"],
    })
result = {"comparison": comparison, "battery_checks": checks}
(root / "cigre_s0_s1_s2_s3_s4_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
pd.DataFrame(comparison).to_csv("results/tables/cigre_s0_s1_s2_s3_s4_comparison.csv", index=False)
der_comparison_figure(summaries, s2, "paper/figures")
print(json.dumps(result, indent=2))
