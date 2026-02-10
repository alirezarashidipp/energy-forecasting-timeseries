#!/usr/bin/env python
"""
Training script for an LSTM model on energy consumption data.

The script reads a CSV file with a `date` and `consumption` column, scales the
target series, creates overlapping sequences of a specified window size, fits
an LSTM network and evaluates its forecasts on a holdout set.
"""
import argparse
import pandas as pd

from src.data.load import load_data
from src.models.lstm import train_lstm, forecast_lstm
from src.eval.metrics import rmse, mae, mape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM model for energy forecasting")
    parser.add_argument("--input-path", required=True, help="Path to the input CSV file")
    parser.add_argument("--test-horizon", type=int, default=14, help="Number of timesteps to forecast for evaluation")
    parser.add_argument("--window-size", type=int, default=7, help="Number of past days used as input sequence")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input_path)
    if 'consumption' not in df.columns:
        raise ValueError("Input CSV must contain a 'consumption' column")
    series = df.sort_values('date')['consumption'].reset_index(drop=True)

    # Train/test split
    test_horizon = args.test_horizon
    train_series = series.iloc[:-test_horizon]
    test_series = series.iloc[-(test_horizon + args.window_size):]  # need extra history for forecasting

    # Train model
    model, scaler = train_lstm(train_series, window_size=args.window_size, epochs=args.epochs, batch_size=args.batch_size)

    # Forecast
    forecasts = forecast_lstm(model, scaler, test_series, window_size=args.window_size, steps=test_horizon)

    y_true = series.iloc[-test_horizon:].values
    y_pred = forecasts

    print("Evaluation on the last {} days:".format(test_horizon))
    print(f"  RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"  MAE:  {mae(y_true, y_pred):.2f}")
    print(f"  MAPE: {mape(y_true, y_pred):.2f}%")


if __name__ == '__main__':
    main()