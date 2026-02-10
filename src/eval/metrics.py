"""
Common evaluation metrics for time series forecasts.

This module includes implementations of RMSE, MAE and MAPE.  Each function
accepts sequences of true and predicted values and returns a scalar score.
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between two sequences.

    Parameters
    ----------
    y_true : array‑like
        Ground truth values.
    y_pred : array‑like
        Predicted values.

    Returns
    -------
    float
        The RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error between two sequences."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error between two sequences."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # avoid division by zero by replacing zeros with a small number
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)