"""
metrics.py — Regression evaluation metrics for solar power prediction.
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute R (Pearson correlation), RMSE, MAE, and sMAPE between
    ground-truth and predicted arrays.

    Parameters
    ----------
    y_true : 1-D array of actual values
    y_pred : 1-D array of predicted values

    Returns
    -------
    dict with keys: R, RMSE, MAE, sMAPE
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64).ravel(), a_min=0, a_max=None)

    # Pearson correlation coefficient
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Symmetric Mean Absolute Percentage Error (avoid division by near-zero)
    # Güneş paneli verilerinde sıfıra yakın (gece/alacakaranlık) değerlerdeki 
    # %200'lük sapmaları önlemek için sadece belirli bir eşik üzerindeki değerler alınır.
    mask = y_true > 5.0
    if mask.sum() == 0:
        smape = float("nan")
    else:
        denominator = np.abs(y_true[mask]) + np.abs(y_pred[mask])
        smape = float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator) * 100)

    return {"R": r, "RMSE": rmse, "MAE": mae, "sMAPE": smape}
