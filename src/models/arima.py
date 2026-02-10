"""
ARIMA model training and forecasting functions.

This module wraps the `statsmodels` ARIMA implementation to provide a simple
API for fitting a model and generating forecasts on a univariate time series.
"""
from typing import Tuple
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_arima(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
    """Fit an ARIMA model to a time series.

    Parameters
    ----------
    series : pandas.Series
        The univariate time series to model.  Missing values should be
        imputed beforehand.
    order : tuple of int, optional
        The (p, d, q) order of the ARIMA model.

    Returns
    -------
    statsmodels.tsa.arima.model.ARIMA
        A fitted ARIMA model.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(model: ARIMA, steps: int = 1) -> pd.Series:
    """Generate out‑of‑sample forecasts from a fitted ARIMA model.

    Parameters
    ----------
    model : statsmodels.tsa.arima.model.ARIMA
        A previously fitted ARIMA model.
    steps : int, optional
        Number of future timesteps to forecast.

    Returns
    -------
    pandas.Series
        A series of forecasted values.
    """
    forecast_result = model.forecast(steps=steps)
    # Ensure we return a pandas Series for consistency
    if not isinstance(forecast_result, pd.Series):
        forecast_result = pd.Series(forecast_result)
    return forecast_result