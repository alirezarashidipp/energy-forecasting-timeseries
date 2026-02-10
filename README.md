# ⚡ Energy Consumption & Price Forecasting

This repository provides a **complete, end‑to‑end framework** for forecasting European electricity demand and prices.  It has been designed to be both a learning resource and a production‑ready codebase.  The project demonstrates how to ingest time‑series data, engineer features, train multiple models (statistical and deep learning), evaluate forecasts with standard metrics and visualise the results.

## 🚀 Why this project?

Energy providers, grid operators and policy makers all rely on accurate forecasts to plan capacity, manage risk and control costs.  As a data scientist you need to demonstrate mastery of classical time‑series methods as well as modern machine‑learning approaches.  This repository showcases:

- **Data pipeline** – modular loaders and pre‑processors that turn raw CSVs into model‑ready datasets.
- **Modelling suite** – implementations for ARIMA, Prophet and LSTM, each encapsulated in its own module.
- **Training scripts** – command line utilities to train and evaluate each model on the same dataset.
- **Evaluation utilities** – reusable functions for RMSE, MAE and MAPE, plus plotting helpers.
- **Tests** – simple unit tests to validate evaluation metrics.

By structuring the code into clear packages (`src/data`, `src/models`, `src/eval`) and providing a sample dataset, you can demonstrate professional software engineering practices while focusing on the data science challenge.

## 🗂 Repository layout

```
energy-forecasting-timeseries/
├── configs/           # YAML configuration files
│   └── prophet.yaml
├── data/
│   ├── raw/          # empty placeholder for raw downloads
│   └── processed/    # sample processed CSV for quick experimentation
├── figures/          # generated plots (empty by default)
├── notebooks/        # Jupyter notebooks for exploratory analysis
├── scripts/          # command line training scripts
├── src/              # core library code
│   ├── data/         # loading and preprocessing utilities
│   ├── eval/         # metrics and plotting helpers
│   └── models/       # ARIMA, Prophet and LSTM implementations
├── tests/            # unit tests
├── requirements.txt  # Python dependencies
├── LICENSE           # Project license (MIT)
└── .gitignore        # Common ignores
```

## 📦 Installation

1. Clone the repository (or copy the folder) and navigate into it:

   ```bash
   git clone https://github.com/USERNAME/energy-forecasting-timeseries.git
   cd energy-forecasting-timeseries
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Inspect the sample data in `data/processed/sample_energy.csv` to familiarise yourself with the format.

## 🧮 Training a model

Each model has its own training script under `scripts/`.  These scripts accept command line arguments for the input file and other hyperparameters and print evaluation metrics on a holdout set.  For example, to train a Prophet model on the sample dataset:

```bash
python scripts/train_prophet.py \
    --input-path data/processed/sample_energy.csv \
    --config-path configs/prophet.yaml \
    --test-horizon 14
```

This command will:

1. Load the CSV and parse the `date` column into a `datetime` object.
2. Split the last 14 days into a test set.
3. Train a Prophet model on the remaining data using the hyperparameters in `configs/prophet.yaml`.
4. Produce forecasts for the next 14 days, compute RMSE, MAE and MAPE against the ground truth and display the results.

Similar scripts exist for ARIMA (`train_arima.py`) and LSTM (`train_lstm.py`).

## 📊 Exploratory notebooks

The `notebooks/` directory contains Jupyter notebooks illustrating how to perform exploratory data analysis and model comparison.  Open them with Jupyter Lab or Notebook and execute the cells:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebooks rely on the same library code under `src/` so you don’t need to duplicate logic.

## 🔧 Extending the project

This framework is intentionally modular.  You can add new models under `src/models` and write corresponding training scripts.  Use the existing modules as a template.  To add new features to the preprocessing pipeline, extend the functions in `src/data/preprocess.py`.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.