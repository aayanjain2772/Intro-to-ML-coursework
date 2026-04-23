# Railways Lagged Linear Regression

## Objective

Predict India railways sector real GDP growth using lagged linear regression features built from the provided annual railway and macro dataset.

## What Was Done

- Cleaned the master railway sheet into a usable annual CSV.
- Created growth-rate features for investment, goods earnings, freight traffic, and passenger kilometres.
- Created one-period lag variables so the regression only uses past information.
- Compared four parsimonious lagged linear regression specifications on a small holdout window.

## Selected Model

- Best holdout model: `demand_mix`
- Predictors: lag_rail_gdp_growth, lag_goods_earnings_growth, lag_freight_growth
- Adjusted R-squared: 0.703
- In-sample RMSE: 1.604

## Forecast Horizons

- Short term (1 year, 2024-25): 11.78%
- Medium term (3 years, 2026-27): 11.03%
- Medium term (5 years, 2028-29): 10.60%
- Long run (10 years, 2033-34): 10.18%

## Forecast

Using the selected lagged linear regression, the predicted railways real GDP growth for 2024-25 is 11.78%.

Longer-horizon forecasts are generated recursively under a baseline assumption that goods-earnings growth and freight growth stay near their recent 3-year averages. Optimistic and pessimistic scenario paths are also saved in `outputs/scenario_forecast_path.csv`.

## Other Models

- Lagged linear regression should remain the main model.
- Ridge or Lasso can be added as a robustness check if you want one extra class-friendly comparison.
- Tree-based models, KNN, and more flexible ML models are not a good idea with this small annual sample.

## Output Files

- `outputs/eda_summary.csv`
- `outputs/missingness_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/final_model_coefficients.csv`
- `outputs/final_model_metrics.csv`
- `outputs/predictions.csv`
- `outputs/horizon_forecasts.csv`
- `outputs/scenario_forecast_path.csv`
- `outputs/model_suitability.csv`
- `outputs/eda_timeseries.svg`
- `outputs/actual_vs_predicted.svg`