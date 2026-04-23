from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "railways_master.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def next_year_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def make_growth(series: pd.Series) -> pd.Series:
    return series.pct_change(fill_method=None) * 100


df = pd.read_csv(DATA_PATH)
df["start_year"] = df["year_label"].str[:4].astype(int)
df = df.sort_values("start_year").reset_index(drop=True)

# Growth transformations for level variables.
df["investment_growth"] = make_growth(df["total_investment_rs_cr"])
df["goods_earnings_growth"] = make_growth(df["goods_earnings_rs_cr"])
df["freight_growth"] = make_growth(df["total_revenue_traffic_tonnes_originating"])
df["passenger_km_growth"] = make_growth(df["passenger_kilometres_million"])

# One-period lags for forecasting.
lag_sources = {
    "lag_rail_gdp_growth": "real_gdp_growth_of_railways",
    "lag_investment_growth": "investment_growth",
    "lag_goods_earnings_growth": "goods_earnings_growth",
    "lag_freight_growth": "freight_growth",
    "lag_passenger_km_growth": "passenger_km_growth",
    "lag_fiscal_deficit": "fiscal_deficit_gdp",
    "lag_real_interest_rate": "real_interest_rate",
    "lag_gdp_deflator": "gdp_deflator_growth_rate_y_y_chg",
}
for lag_name, source in lag_sources.items():
    df[lag_name] = df[source].shift(1)

candidate_predictors = [
    "lag_rail_gdp_growth",
    "lag_investment_growth",
    "lag_goods_earnings_growth",
    "lag_freight_growth",
    "lag_passenger_km_growth",
    "lag_fiscal_deficit",
    "lag_real_interest_rate",
    "lag_gdp_deflator",
]

lasso_data = df[["year_label", "start_year", "real_gdp_growth_of_railways"] + candidate_predictors].dropna().reset_index(drop=True)

# Keep the last 3 usable observations for holdout evaluation.
train_data = lasso_data.iloc[:-3].copy()
test_data = lasso_data.iloc[-3:].copy()

X_train = train_data[candidate_predictors]
y_train = train_data["real_gdp_growth_of_railways"]
X_test = test_data[candidate_predictors]
y_test = test_data["real_gdp_growth_of_railways"]

cv_folds = min(5, len(train_data))
lasso_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv_folds, random_state=42, max_iter=100000)),
    ]
)
lasso_pipe.fit(X_train, y_train)

lasso = lasso_pipe.named_steps["lasso"]
selected_variables = [
    feature for feature, coef in zip(candidate_predictors, lasso.coef_) if abs(coef) > 1e-8
]
if not selected_variables:
    selected_variables = ["lag_rail_gdp_growth"]

selected_rows = [
    {
        "variable": feature,
        "lasso_coefficient": float(coef),
        "selected": "yes" if feature in selected_variables else "no",
    }
    for feature, coef in zip(candidate_predictors, lasso.coef_)
]
pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "lasso_selected_variables.csv", index=False)

# Refit plain linear regression on selected variables for final interpretation.
selected_model_data = df[["year_label", "start_year", "real_gdp_growth_of_railways"] + selected_variables].dropna().reset_index(drop=True)
selected_train = selected_model_data.iloc[:-3].copy()
selected_test = selected_model_data.iloc[-3:].copy()

ols = LinearRegression()
ols.fit(selected_train[selected_variables], selected_train["real_gdp_growth_of_railways"])

train_pred = ols.predict(selected_train[selected_variables])
test_pred = ols.predict(selected_test[selected_variables])

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


model_summary = pd.DataFrame(
    [
        {
            "selection_method": "lasso_cv_then_ols",
            "selected_variables": ", ".join(selected_variables),
            "train_observations": len(selected_train),
            "test_observations": len(selected_test),
            "lasso_alpha": float(lasso.alpha_),
            "train_rmse": rmse(selected_train["real_gdp_growth_of_railways"].to_numpy(), train_pred),
            "test_rmse": rmse(selected_test["real_gdp_growth_of_railways"].to_numpy(), test_pred),
            "test_mae": mae(selected_test["real_gdp_growth_of_railways"].to_numpy(), test_pred),
            "r_squared_full_sample": float(ols.score(selected_model_data[selected_variables], selected_model_data["real_gdp_growth_of_railways"])),
        }
    ]
)
model_summary.to_csv(OUTPUT_DIR / "lasso_model_summary.csv", index=False)

coefficients = pd.DataFrame(
    [{"term": "intercept", "estimate": float(ols.intercept_)}]
    + [{"term": variable, "estimate": float(coef)} for variable, coef in zip(selected_variables, ols.coef_)]
)
coefficients.to_csv(OUTPUT_DIR / "lasso_final_coefficients.csv", index=False)

predictions = selected_model_data[["year_label", "start_year", "real_gdp_growth_of_railways"]].copy()
predictions["fitted"] = ols.predict(selected_model_data[selected_variables])
predictions["residual"] = predictions["real_gdp_growth_of_railways"] - predictions["fitted"]
predictions.to_csv(OUTPUT_DIR / "lasso_predictions.csv", index=False)

# One-year forecast uses actual 2024-25 lagged inputs.
forecast_row = df.loc[df["year_label"] == "2024-25", ["year_label", "start_year"] + selected_variables].dropna()
if forecast_row.empty:
    raise RuntimeError("Could not generate 2024-25 forecast because required lagged predictors are missing.")

one_year_forecast = float(ols.predict(forecast_row[selected_variables])[0])
pd.DataFrame(
    [{"forecast_year": "2024-25", "predicted_real_gdp_growth_of_railways": one_year_forecast}]
).to_csv(OUTPUT_DIR / "lasso_forecast_2024_25.csv", index=False)

# Build recursive horizon forecasts under baseline/optimistic/pessimistic assumptions.
assumption_vars = [var for var in selected_variables if var != "lag_rail_gdp_growth"]
assumption_map = {}
for var in assumption_vars:
    base_var = var.replace("lag_", "")
    series = df[base_var].dropna()
    baseline = float(series.tail(3).mean())
    sd = float(series.std(ddof=0))
    assumption_map[var] = {
        "baseline": baseline,
        "optimistic": baseline + 0.5 * sd,
        "pessimistic": baseline - 0.5 * sd,
    }

scenario_rows = []
for scenario in ["baseline", "optimistic", "pessimistic"]:
    prev_growth = one_year_forecast
    for horizon in range(1, 11):
        forecast_start_year = 2023 + horizon
        if horizon == 1:
            predicted_growth = one_year_forecast
        else:
            x = {}
            for variable in selected_variables:
                if variable == "lag_rail_gdp_growth":
                    x[variable] = prev_growth
                else:
                    x[variable] = assumption_map[variable][scenario]
            predicted_growth = float(ols.predict(pd.DataFrame([x]))[0])
        scenario_rows.append(
            {
                "scenario": scenario,
                "horizon_years_ahead": horizon,
                "forecast_year": next_year_label(forecast_start_year),
                "predicted_real_gdp_growth_of_railways": predicted_growth,
            }
        )
        prev_growth = predicted_growth

scenario_df = pd.DataFrame(scenario_rows)
scenario_df.to_csv(OUTPUT_DIR / "lasso_scenario_forecast_path.csv", index=False)

baseline_path = scenario_df.loc[scenario_df["scenario"] == "baseline"].reset_index(drop=True)
horizon_df = pd.DataFrame(
    [
        {"horizon": "short_term_1_year", "forecast_year": baseline_path.loc[0, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[0, "predicted_real_gdp_growth_of_railways"]},
        {"horizon": "medium_term_3_year", "forecast_year": baseline_path.loc[2, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[2, "predicted_real_gdp_growth_of_railways"]},
        {"horizon": "medium_term_5_year", "forecast_year": baseline_path.loc[4, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[4, "predicted_real_gdp_growth_of_railways"]},
        {"horizon": "long_run_10_year", "forecast_year": baseline_path.loc[9, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[9, "predicted_real_gdp_growth_of_railways"]},
    ]
)
horizon_df.to_csv(OUTPUT_DIR / "lasso_horizon_forecasts.csv", index=False)

report_lines = [
    "# Lasso-Based Variable Selection Report",
    "",
    "## Why Lasso Was Used",
    "",
    "The dataset contains many possible railway and macro predictors but only a small annual sample. Lasso is a class-covered shrinkage method that helps reduce the predictor set in a formal way by shrinking weak coefficients toward zero.",
    "",
    "## Candidate Lagged Variables",
    "",
    "- lag_rail_gdp_growth",
    "- lag_investment_growth",
    "- lag_goods_earnings_growth",
    "- lag_freight_growth",
    "- lag_passenger_km_growth",
    "- lag_fiscal_deficit",
    "- lag_real_interest_rate",
    "- lag_gdp_deflator",
    "",
    "## Selected Variables",
    "",
    f"Lasso selected: {', '.join(selected_variables)}",
    "",
    "## Final Forecasts",
    "",
    f"- 1 year: {horizon_df.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 3 years: {horizon_df.loc[1, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 5 years: {horizon_df.loc[2, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 10 years: {horizon_df.loc[3, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    "",
    "## Notes",
    "",
    "After Lasso selected the variables, a plain linear regression was refit on the selected set so that the final model remains easy to interpret and explain in the report.",
]
(DOCS_DIR / "lasso_report.md").write_text("\n".join(report_lines))

print("Selected variables:", ", ".join(selected_variables))
