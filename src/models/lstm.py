"""
LSTM model utilities for time series forecasting.

This module provides helper functions to prepare sequences, build and train
an LSTM network using TensorFlow/Keras, and generate future forecasts.  The
model operates on a univariate target series.
"""
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def create_sequences(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping sequences and targets from a 1D array.

    Parameters
    ----------
    series : ndarray
        One‑dimensional array of scaled values.
    window_size : int
        Length of each input sequence.

    Returns
    -------
    X : ndarray, shape (n_samples, window_size, 1)
        Array of input sequences.
    y : ndarray, shape (n_samples,)
        Array of target values corresponding to each sequence.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    X = np.array(X)
    y = np.array(y)
    # reshape for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm_model(input_shape: Tuple[int, int], hidden_units: int = 50) -> tf.keras.Model:
    """Construct an LSTM network for regression.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input sequences (timesteps, features).
    hidden_units : int, optional
        Number of units in the LSTM layer.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(hidden_units, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(series: pd.Series, window_size: int = 7, epochs: int = 20, batch_size: int = 16) -> Tuple[tf.keras.Model, MinMaxScaler]:
    """Scale a time series, construct sequences and train an LSTM model.

    Parameters
    ----------
    series : pandas.Series
        Univariate time series to model.
    window_size : int, optional
        Length of each input sequence.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size for training.

    Returns
    -------
    model : tf.keras.Model
        Trained LSTM model.
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler used to normalise the series.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = create_sequences(scaled, window_size)
    model = build_lstm_model((window_size, 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler


def forecast_lstm(model: tf.keras.Model, scaler: MinMaxScaler, series: pd.Series, window_size: int = 7, steps: int = 1) -> np.ndarray:
    """Forecast future values using a trained LSTM model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used to normalise the training data.
    series : pandas.Series
        Original (unscaled) time series containing enough history for forecasting.
    window_size : int, optional
        Length of the window used to generate input sequences.
    steps : int, optional
        Number of future timesteps to forecast.

    Returns
    -------
    numpy.ndarray
        Array of forecasted values in the original scale.
    """
    # Scale the entire series using the existing scaler
    scaled_series = scaler.transform(series.values.reshape(-1, 1)).flatten()
    last_seq = scaled_series[-window_size:].tolist()
    forecasts = []
    for _ in range(steps):
        X_input = np.array(last_seq).reshape((1, window_size, 1))
        pred = model.predict(X_input, verbose=0)[0, 0]
        forecasts.append(pred)
        # slide window: append prediction and drop oldest
        last_seq = last_seq[1:] + [pred]
    forecasts = np.array(forecasts).reshape(-1, 1)
    # invert scaling
    return scaler.inverse_transform(forecasts).flatten()