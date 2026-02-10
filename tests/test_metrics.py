"""
Unit tests for the evaluation metrics defined in src/eval/metrics.py.

These tests can be run with pytest.  They verify that the RMSE, MAE and MAPE
functions return zero for identical sequences and produce expected values for
simple examples.
"""
import numpy as np
from src.eval.metrics import rmse, mae, mape


def test_perfect_predictions() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert mape(y_true, y_pred) == 0.0


def test_nonzero_errors() -> None:
    y_true = [2.0, 4.0, 6.0]
    y_pred = [3.0, 5.0, 7.0]
    assert np.isclose(rmse(y_true, y_pred), 1.0)
    assert np.isclose(mae(y_true, y_pred), 1.0)
    assert np.isclose(mape(y_true, y_pred), (1/2 + 1/4 + 1/6) / 3 * 100)