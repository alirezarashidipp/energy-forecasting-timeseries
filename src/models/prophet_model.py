"""
Prophet model training and forecasting utilities.

This module wraps the `prophet` package (also known as fbprophet) to provide
simple functions for fitting a model and obtaining future forecasts.  The
functions accept configuration dictionaries loaded from YAML files.
"""
from typing import Dict, Optional
import pandas as pd
from prophet import Prophet


def train_prophet(df: pd.DataFrame, config: Optional[Dict] = None) -> Prophet:
    """Train a Prophet model on a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``ds`` (dates) and ``y`` (target).
    config : dict, optional
        Dictionary of hyperparameters to pass to the Prophet constructor.

    Returns
    -------
    prophet.Prophet
        A fitted Prophet model.
    """
    model = Prophet(**(config or {}))
    model.fit(df)
    return model


def forecast_prophet(model: Prophet, periods: int, freq: str = 'D') -> pd.DataFrame:
    """Generate forecasts from a trained Prophet model.

    Parameters
    ----------
    model : prophet.Prophet
        A fitted Prophet model.
    periods : int
        Number of periods to forecast into the future.
    freq : str, optional
        Frequency of the future dataframe (default daily).  Accepts any
        pandas offset alias (e.g. 'H' for hourly).

    Returns
    -------
    pandas.DataFrame
        Prophet forecast with columns including ``ds`` and ``yhat``.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]