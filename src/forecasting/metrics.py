from __future__ import annotations

import numpy as np


def pinball(y, prediction, quantile: float) -> float:
    error = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def point_metrics(y, prediction, mase_scale=None) -> dict:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = y - prediction
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = np.maximum(np.abs(y) + np.abs(prediction), 1e-12)
    result = {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "nmae_percent_of_mean": float(100 * mae / np.mean(np.abs(y))),
        "smape_percent": float(200 * np.mean(np.abs(error) / denominator)),
    }
    if mase_scale is not None:
        result["mase"] = float(mae / mase_scale)
    return result


def interval_metrics(y, lower, upper, alpha=0.2) -> dict:
    y, lower, upper = map(lambda z: np.asarray(z, dtype=float), (y, lower, upper))
    width = upper - lower
    score = width.copy()
    score[y < lower] += (2 / alpha) * (lower[y < lower] - y[y < lower])
    score[y > upper] += (2 / alpha) * (y[y > upper] - upper[y > upper])
    return {
        "picp_percent": float(100 * np.mean((y >= lower) & (y <= upper))),
        "pinaw_percent_of_range": float(100 * np.mean(width) / (np.max(y) - np.min(y))),
        "mean_winkler_score_kw": float(np.mean(score)),
    }

