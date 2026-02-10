"""
Functions for preprocessing time series data prior to modelling.

These utilities handle missing value imputation, creation of date‑based
features, lagged features and rolling window statistics.  All functions
return new DataFrames rather than mutating their inputs.
"""
from typing import Iterable, List
import pandas as pd


def impute_missing(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with potential missing values.
    method : str, optional
        Imputation strategy: 'ffill' (forward fill) or 'bfill' (backward fill).

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values imputed.
    """
    if method not in {'ffill', 'bfill'}:
        raise ValueError("method must be 'ffill' or 'bfill'")
    return df.fillna(method=method)


def add_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Add year, month and day‑of‑week features to a DataFrame.

    The input DataFrame must contain a datetime column.  The function
    returns a copy with additional columns for year, month and day of week.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a date column.
    date_col : str, optional
        Name of the date column to expand.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added date features.
    """
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in DataFrame")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    return df


def create_lag_features(df: pd.DataFrame, column: str, lags: Iterable[int]) -> pd.DataFrame:
    """Add lagged versions of a column to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Name of the column to lag.
    lags : Iterable[int]
        Sequence of lag periods (number of timesteps).

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional lagged columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, column: str, windows: Iterable[int]) -> pd.DataFrame:
    """Add rolling mean features for a column to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Column on which to compute rolling means.
    windows : Iterable[int]
        Sequence of window sizes (number of timesteps).

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional rolling mean columns.
    """
    df = df.copy()
    for window in windows:
        df[f"{column}_rollmean_{window}"] = df[column].rolling(window).mean()
    return df