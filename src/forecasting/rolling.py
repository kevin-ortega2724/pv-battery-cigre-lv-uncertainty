from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import make_supervised
from .metrics import interval_metrics, pinball, point_metrics


def rolling_origin_2009(processed_csv: str | Path, output_path: str | Path,
                        horizon_steps: int = 4, seed: int = 20260904) -> dict:
    data = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
    frame = make_supervised(data["Global_active_power"], horizon_steps)
    features = [c for c in frame if c != "target"]
    folds = [
        ("2009-01-01", "2009-03-31 23:59:59"),
        ("2009-04-01", "2009-06-30 23:59:59"),
        ("2009-07-01", "2009-09-30 23:59:59"),
        ("2009-10-01", "2009-12-31 23:59:59"),
    ]
    all_predictions = []
    fold_results = []
    params = dict(max_iter=140, learning_rate=.06, max_leaf_nodes=31,
                  min_samples_leaf=30, l2_regularization=.1, random_state=seed)
    for number, (start, end) in enumerate(folds, 1):
        validation = frame.loc[start:end]
        train = frame.loc[: pd.Timestamp(start) - pd.Timedelta(seconds=1)]
        prediction = pd.DataFrame(index=validation.index)
        prediction["observed_kw"] = validation["target"]
        for q in (.1, .5, .9):
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **params)
            model.fit(train[features], train["target"])
            prediction[f"q{int(q*100):02d}_kw"] = model.predict(validation[features])
        prediction[["q10_kw", "q50_kw", "q90_kw"]] = np.sort(prediction[["q10_kw", "q50_kw", "q90_kw"]], axis=1)
        metrics = point_metrics(prediction["observed_kw"], prediction["q50_kw"])
        metrics.update(interval_metrics(prediction["observed_kw"], prediction["q10_kw"], prediction["q90_kw"]))
        metrics.update({"fold": number, "start": start, "end": end,
                        "train_rows": len(train), "validation_rows": len(validation)})
        fold_results.append(metrics)
        prediction["fold"] = number
        all_predictions.append(prediction)
    combined = pd.concat(all_predictions)
    aggregate = point_metrics(combined["observed_kw"], combined["q50_kw"])
    aggregate.update(interval_metrics(combined["observed_kw"], combined["q10_kw"], combined["q90_kw"]))
    aggregate.update({f"pinball_q{int(q*100):02d}_kw": pinball(combined["observed_kw"], combined[f"q{int(q*100):02d}_kw"], q) for q in (.1,.5,.9)})
    result = {"horizon_steps": horizon_steps, "folds": fold_results, "aggregate": aggregate,
              "note": "Expanding-window quarterly rolling-origin evaluation confined to 2009; 2010 remains untouched."}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    combined.to_csv(path.with_suffix(".csv"), index_label="issue_time")
    return result

