![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5-green.svg)

# ⚡ Energy Consumption & Price Forecasting (Time Series)

This project develops a
robust framework for **time series forecasting** of European energy consumption and pricing.  
It compares **statistical models (ARIMA, Prophet)** with **deep learning (LSTM)** to evaluate forecasting accuracy and applicability in real-world energy planning.

---

## 📊 Project Overview
- **Domain:** Energy Analytics / Time Series Forecasting  
- **Objective:** Predict daily electricity consumption and prices  
- **Impact:** Support energy providers and policymakers in demand planning, cost optimization, and sustainability strategies  

---

## 🗂 Data
- Source: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) (European energy data)  
- Frequency: Daily consumption & market prices  
- Preprocessing:
  - Missing value imputation
  - Lag & rolling window features
  - Calendar-based features (day of week, holidays, seasonality)

---

## 🧮 Models Implemented
1. **ARIMA** – Baseline statistical approach  
2. **Prophet (Meta/Facebook)** – Trend & seasonality decomposition  
3. **LSTM (Keras/TensorFlow)** – Deep learning for sequential dependencies  

---

## 🎯 Results
- **Evaluation Metrics:** RMSE, MAE, MAPE  
- **Key Insights:**
  - Prophet captured strong seasonality patterns  
  - LSTM outperformed ARIMA in capturing non-linear dependencies  
  - Combining models reduced overall forecasting error  

*(Detailed results with plots available in `/notebooks`)*  

---

## 🚀 Future Work
- Streamlit dashboard for interactive forecasting  
- Integration with real-time API (ENTSO-E live data)  
- Model explainability using SHAP for time series  

---

## 🛠 Tech Stack
- Python (Pandas, NumPy, Scikit-learn)  
- Statsmodels (ARIMA)  
- Prophet  
- TensorFlow / Keras (LSTM)  
- Matplotlib, Seaborn, Plotly (visualization)  

---

## 📂 Repository Structure
energy-forecasting-timeseries/
├── data/ # Raw & processed datasets
├── notebooks/ # Jupyter notebooks (EDA, models, evaluation)
├── src/ # Python modules for preprocessing & training
├── figures/ # Plots and visualizations
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## ▶️ How to Run
```bash
git clone https://github.com/USERNAME/energy-forecasting-timeseries.git
cd energy-forecasting-timeseries
pip install -r requirements.txt
jupyter notebook



Author
Ali Reza Rashidi
Data Scientist & BI Engineer | 9+ years in ML, NLP & BI










energy-forecasting-timeseries/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── prophet.yaml
├── data/
│   ├── raw/        # empty/.gitkeep
│   └── processed/  # sample daily.csv
├── figures/
│   └── prophet_forecast.png
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_prophet_baseline.ipynb
├── outputs/        # ignored by git
├── scripts/
│   └── train_prophet.py
├── src/
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── models/
│   │   └── prophet.py
│   └── eval/
│       ├── metrics.py
│       └── plots.py
└── tests/
    └── test_metrics.py
