from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import make_supervised, temporal_split
from .metrics import interval_metrics, pinball, point_metrics


def run_pilot(processed_csv: str | Path, output_dir: str | Path, horizon_steps: int = 4, seed: int = 20260904) -> dict:
    data = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
    supervised = make_supervised(data["Global_active_power"], horizon_steps)
    train, validation, test = temporal_split(supervised)
    fit = pd.concat([train, validation])
    features = [column for column in supervised.columns if column != "target"]
    x_fit, y_fit = fit[features], fit["target"]
    x_test, y_test = test[features], test["target"]

    predictions = pd.DataFrame(index=test.index)
    predictions["observed_kw"] = y_test
    daily_lag = 96 - horizon_steps
    predictions["daily_persistence_kw"] = (
        test["last_observed"] if daily_lag == 0 else test[f"lag_{daily_lag}"]
    )
    predictions["weekly_persistence_kw"] = test[f"lag_{672 - horizon_steps}"]

    params = dict(max_iter=180, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=30, l2_regularization=0.1, random_state=seed)
    validation_predictions = {}
    for q in (0.1, 0.5, 0.9):
        calibration_model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **params)
        calibration_model.fit(train[features], train["target"])
        validation_predictions[q] = calibration_model.predict(validation[features])
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **params)
        model.fit(x_fit, y_fit)
        predictions[f"q{int(q * 100):02d}_kw"] = model.predict(x_test)

    # Enforce ordered reported quantiles without using outcomes.
    qcols = ["q10_kw", "q50_kw", "q90_kw"]
    predictions[qcols] = np.sort(predictions[qcols].to_numpy(), axis=1)
    validation_ordered = np.sort(np.column_stack([validation_predictions[q] for q in (0.1, 0.5, 0.9)]), axis=1)
    conformity = np.maximum(validation_ordered[:, 0] - validation["target"].to_numpy(),
                            validation["target"].to_numpy() - validation_ordered[:, 2])
    alpha = 0.2
    level = min(1.0, np.ceil((len(conformity) + 1) * (1 - alpha)) / len(conformity))
    qhat = float(np.quantile(conformity, level, method="higher"))
    predictions["q10_calibrated_kw"] = predictions["q10_kw"] - qhat
    predictions["q90_calibrated_kw"] = predictions["q90_kw"] + qhat
    mase_scale = float(np.mean(np.abs(np.diff(train["target"].to_numpy()))))
    result = {
        "horizon_steps": horizon_steps,
        "horizon_hours": horizon_steps / 4,
        "strategy": "direct",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": len(features),
        "median_quantile_model": point_metrics(y_test, predictions["q50_kw"], mase_scale),
        "daily_persistence": point_metrics(y_test, predictions["daily_persistence_kw"], mase_scale),
        "weekly_persistence": point_metrics(y_test, predictions["weekly_persistence_kw"], mase_scale),
        "probabilistic": {
            "pinball_q10_kw": pinball(y_test, predictions["q10_kw"], 0.1),
            "pinball_q50_kw": pinball(y_test, predictions["q50_kw"], 0.5),
            "pinball_q90_kw": pinball(y_test, predictions["q90_kw"], 0.9),
            **interval_metrics(y_test, predictions["q10_kw"], predictions["q90_kw"]),
        },
        "conformal_calibration": {
            "validation_qhat_kw": qhat,
            **interval_metrics(y_test, predictions["q10_calibrated_kw"], predictions["q90_calibrated_kw"]),
        },
        "leakage_control": "Load history ends at issue time; target-aligned calendar is known in advance.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / f"pilot_predictions_h{horizon_steps}.csv", index_label="issue_time")
    (output / f"pilot_metrics_h{horizon_steps}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
