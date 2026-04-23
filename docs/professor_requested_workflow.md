# Professor-Requested Workflow

## Structure

1. Lasso for variable selection
2. PCA on the selected macro block
3. Current-variable model for 2024-25 nowcasting
4. Lagged-variable model for medium-run and long-run forecasting

## Current-Variable Model

- Lasso selected: operating_ratio, fiscal_deficit_gdp
- Final predictors after PCA step: operating_ratio, fiscal_deficit_gdp
- Predicted 2024-25 railways GDP growth: 8.04%
- PCA note: After Lasso, there were not enough selected macro variables to form principal components, so the selected macro variable was kept directly.

## Lagged Forecasting Model

- Lasso selected: lag_rail_gdp_growth, lag_operating_ratio
- Final predictors after PCA step: lag_rail_gdp_growth, lag_operating_ratio
- 1-year forecast: 11.71%
- 5-year forecast: 11.76%
- 10-year forecast: 11.77%
- PCA note: After Lasso, there were not enough selected lagged macro variables to form principal components, so PCA did not add extra components here.

## Interpretation

The current-variable model is best described as a nowcasting or contemporaneous prediction model because it uses variables from the same year to explain 2024-25 railways GDP growth. The lagged-variable model is the true forecasting model because it uses only past information.