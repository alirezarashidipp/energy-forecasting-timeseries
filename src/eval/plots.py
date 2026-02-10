"""
Plotting helpers for visualising forecasts against actual data.

These functions utilise matplotlib to produce common diagnostic plots.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(dates, actual, forecast, title: str = 'Forecast vs Actual') -> plt.Figure:
    """Plot actual and forecasted values on a shared time axis.

    Parameters
    ----------
    dates : array‑like
        Sequence of dates corresponding to the observations.
    actual : array‑like
        True values.
    forecast : array‑like
        Predicted values.
    title : str, optional
        Title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, label='Actual', marker='o')
    ax.plot(dates, forecast, label='Forecast', marker='x')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig