"""
Utility functions for loading and saving datasets used in the energy forecasting
project.  These helpers wrap pandas I/O and ensure that date columns are
parsed correctly.
"""
from pathlib import Path
from typing import Union
import pandas as pd


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    If the file contains a column named ``date``, it will be parsed into
    ``datetime64`` dtype.  Other columns are left unchanged.

    Parameters
    ----------
    path: str or Path
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def save_data(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a pandas DataFrame to CSV.

    The parent directories are created if they do not exist.  The index is
    omitted from the output file.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to save.
    path: str or Path
        Path where the CSV should be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)