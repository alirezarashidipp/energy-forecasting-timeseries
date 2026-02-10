#!/usr/bin/env python
"""
Simple training script for an ARIMA model on energy consumption data.

The script reads a CSV with `date` and `consumption` columns, trains an ARIMA
model on the training portion of the series and evaluates it on a holdout set.
The ARIMA `p`, `d`, `q` order can be customised via command line flags.
"""
import argparse
import pandas as pd

from src.data.load import load_data
from src.models.arima import train_arima, forecast_arima
from src.eval.metrics import rmse, mae, mape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ARIMA model for energy forecasting")
    parser.add_argument("--input-path", required=True, help="Path to input CSV file")
    parser.add_argument("--test-horizon", type=int, default=14, help="Number of days to hold out for testing")
    parser.add_argument("--p", type=int, default=1, help="ARIMA p parameter (autoregressive order)")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d parameter (difference order)")
    parser.add_argument("--q", type=int, default=1, help="ARIMA q parameter (moving average order)")
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
    test_series = series.iloc[-test_horizon:]

    # Train ARIMA
    model = train_arima(train_series, order=(args.p, args.d, args.q))

    # Forecast future values
    forecasts = forecast_arima(model, steps=test_horizon)

    y_true = test_series.values
    y_pred = forecasts.values

    print("Evaluation on the last {} days:".format(test_horizon))
    print(f"  RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"  MAE:  {mae(y_true, y_pred):.2f}")
    print(f"  MAPE: {mape(y_true, y_pred):.2f}%")


if __name__ == '__main__':
    main()