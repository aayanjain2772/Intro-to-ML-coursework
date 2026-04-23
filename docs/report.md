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

## Forecast

Using the selected lagged linear regression, the predicted railways real GDP growth for 2024-25 is 11.78%.

## Output Files

- `outputs/eda_summary.csv`
- `outputs/missingness_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/final_model_coefficients.csv`
- `outputs/final_model_metrics.csv`
- `outputs/predictions.csv`
- `outputs/eda_timeseries.svg`
- `outputs/actual_vs_predicted.svg`