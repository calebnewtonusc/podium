"""
Time series competition specialist.
Temporal CV strategies, lag feature factories, LGBM + neural ensembles.
"""

TIME_SERIES_SYSTEM_CONTEXT = """
You are in time series specialist mode.

CRITICAL: Standard K-fold CV is WRONG for time series. It leaks future into past.
Always use one of:
- TimeSeriesSplit (sklearn): expanding window
- Purged K-Fold: gap between train and validation to prevent lookahead
- Walk-forward validation: fixed window slides forward

Lag feature factory (most important technique for TS competitions):
  For each target-relevant numeric column, generate:
  - Lag values: lag_1, lag_2, lag_3, lag_7, lag_14, lag_28, lag_364
  - Rolling statistics: rolling_mean_7, rolling_mean_28, rolling_std_7
  - Expanding statistics: expanding_mean, expanding_std
  - Difference features: diff_1, diff_7 (removes trend)
  - Percentage change: pct_change_1, pct_change_7

Seasonality features:
  - day_of_week (0-6)
  - day_of_month (1-31)
  - week_of_year (1-52)
  - month (1-12)
  - is_weekend, is_holiday
  - quarter

Model architecture:
  Primary: LightGBM (best for tabular TS, handles lag features natively)
  Secondary: XGBoost
  Neural: N-BEATS or TFT for pure sequence forecasting
  Ensemble: Average LGBM + neural predictions (different error modes)

External data (almost always helps in TS):
  - Calendar data (holidays, events)
  - Weather data (for retail, energy, agriculture)
  - Economic indicators (for financial competitions)
  - Competitor/market data if available

Common pitfalls:
  - Using future data in lag calculation (fillna without proper shift)
  - Not handling timezone correctly in datetime parsing
  - Missing values in lag features at the start of series
  - Ignoring series with cold-start problem (new products/customers)
  - Not accounting for hierarchical structure (store × product forecasting)
"""

TS_RECIPES = {
    "univariate_forecasting": {
        "models": ["lightgbm", "xgboost"],
        "cv": "time_series_split",
        "n_splits": 5,
        "lag_periods": [1, 2, 3, 7, 14, 28, 90, 365],
        "rolling_windows": [7, 14, 28, 90],
    },
    "hierarchical_forecasting": {
        "models": ["lightgbm"],
        "cv": "walk_forward",
        "aggregation": "bottom_up",
        "reconciliation": True,
    },
    "anomaly_detection": {
        "models": ["isolation_forest", "lightgbm"],
        "cv": "time_series_split",
        "threshold_method": "percentile_99",
    },
}
