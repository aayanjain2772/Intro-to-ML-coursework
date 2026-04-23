# Lasso-Based Variable Selection Report

## Why Lasso Was Used

The dataset contains many possible railway and macro predictors but only a small annual sample. Lasso is a class-covered shrinkage method that helps reduce the predictor set in a formal way by shrinking weak coefficients toward zero.

## Candidate Lagged Variables

- lag_rail_gdp_growth
- lag_investment_growth
- lag_goods_earnings_growth
- lag_freight_growth
- lag_passenger_km_growth
- lag_fiscal_deficit
- lag_real_interest_rate
- lag_gdp_deflator

## Selected Variables

Lasso selected: lag_rail_gdp_growth

## Final Forecasts

- 1 year: 11.64%
- 3 years: 11.70%
- 5 years: 11.74%
- 10 years: 11.83%

## Notes

After Lasso selected the variables, a plain linear regression was refit on the selected set so that the final model remains easy to interpret and explain in the report.