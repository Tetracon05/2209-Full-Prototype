"""
metrics.py — Regression evaluation metrics for solar power prediction.
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute R (Pearson correlation), RMSE, MAE, and MAPE between
    ground-truth and predicted arrays.

    Parameters
    ----------
    y_true : 1-D array of actual values
    y_pred : 1-D array of predicted values

    Returns
    -------
    dict with keys: R, RMSE, MAE, MAPE
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Pearson correlation coefficient
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {"R": r, "RMSE": rmse, "MAE": mae, "MAPE": mape}
