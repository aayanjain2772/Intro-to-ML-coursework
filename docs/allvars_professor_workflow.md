# All-Variable Professor Workflow

This workflow lets Lasso see the full cleaned numeric dataset rather than a hand-curated candidate subset.

## Current-Variable Branch

- Candidate variables given to Lasso: 60
- Lasso selected count: 14
- Lasso selected variables: total_route_length, number_of_stations, no_of_locomotive_in_service_steam, total_iron_ore_net_tonne_kilometres, food_grains_tonnes_originating, total_container_net_tonne_kilometres, passengers_originating_million, average_number_of_trains_run_daily, no_of_employees_in_thousands, fiscal_deficit_gdp, real_interest_rate, gdp_deflator_growth_rate_y_y_chg, energy_price_inflation_y_y_chg, contribution_to_real_gdp_railways_disc_pt
- PCA features used in final model: PC1, PC2, PC3, PC4, PC5, PC6
- 2024-25 current-year nowcast: 11.96%

## Lagged Branch

- Candidate lagged variables given to Lasso: 61
- Lasso selected count: 7
- Lasso selected variables: lag_electrified_route_kilometres_by_line, lag_capital_at_charge_rs_cr, lag_operating_ratio, lag_total_iron_ore_net_tonne_kilometres, lag_passenger_kilometres_million, lag_no_of_employees_in_thousands, lag_gdp_deflator_growth_rate_y_y_chg
- PCA features used in final lagged model: PC1, PC2, PC3, PC4
- 2024-25 lagged forecast: 14.91%
- 5-year lagged forecast: 16.10%
- 10-year lagged forecast: 16.10%