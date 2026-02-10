#!/usr/bin/env python
"""
Command line script to train a Prophet model on a given energy dataset.

The script loads a CSV file containing a `date` column and a target column named
`consumption`, splits the data into a training and test set, fits a Prophet
model using hyperparameters provided in a YAML configuration file and prints
evaluation metrics on the test set.

Usage example:

    python scripts/train_prophet.py \
        --input-path data/processed/sample_energy.csv \
        --config-path configs/prophet.yaml \
        --test-horizon 14

This will train on all but the last 14 observations and evaluate on the last
14 days of the dataset.
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path

from src.data.load import load_data
from src.models.prophet_model import train_prophet, forecast_prophet
from src.eval.metrics import rmse, mae, mape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Prophet model for energy forecasting")
    parser.add_argument("--input-path", required=True, help="Path to the CSV file containing the time series")
    parser.add_argument("--config-path", default="configs/prophet.yaml", help="Path to YAML file with Prophet hyperparameters")
    parser.add_argument("--test-horizon", type=int, default=14, help="Number of days to hold out for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load dataset
    df = load_data(args.input_path)
    if 'consumption' not in df.columns:
        raise ValueError("Input CSV must contain a 'consumption' column")
    df = df.sort_values('date').reset_index(drop=True)

    # Split into train/test
    test_horizon = args.test_horizon
    train_df = df.iloc[:-test_horizon].copy()
    test_df = df.iloc[-test_horizon:].copy()

    # Load Prophet configuration
    if args.config_path and Path(args.config_path).exists():
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Prepare data for Prophet: rename columns
    train_prophet_df = train_df.rename(columns={'date': 'ds', 'consumption': 'y'})

    # Train model
    model = train_prophet(train_prophet_df, config=config)

    # Forecast future values
    forecast_df = forecast_prophet(model, periods=test_horizon)

    # Align forecast with test set
    # Prophet returns forecasts for the entire training + future period; we only need the last `test_horizon` rows
    y_pred = forecast_df['yhat'].tail(test_horizon).values
    y_true = test_df['consumption'].values

    print("Evaluation on the last {} days:".format(test_horizon))
    print(f"  RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"  MAE:  {mae(y_true, y_pred):.2f}")
    print(f"  MAPE: {mape(y_true, y_pred):.2f}%")


if __name__ == "__main__":
    main()