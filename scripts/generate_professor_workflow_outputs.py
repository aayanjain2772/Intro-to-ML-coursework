from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def run_lasso_selection(data: pd.DataFrame, target: str, predictors: list[str]) -> tuple[pd.DataFrame, list[str], Pipeline]:
    usable = data[[target] + predictors].dropna().copy()
    X = usable[predictors]
    y = usable[target]
    cv_folds = min(5, len(usable))
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=cv_folds, random_state=42, max_iter=100000)),
        ]
    )
    pipe.fit(X, y)
    lasso = pipe.named_steps["lasso"]
    selection = pd.DataFrame(
        {
            "variable": predictors,
            "lasso_coefficient": lasso.coef_,
            "selected": ["yes" if abs(coef) > 1e-8 else "no" for coef in lasso.coef_],
        }
    )
    selected = selection.loc[selection["selected"] == "yes", "variable"].tolist()
    return selection, selected, pipe


def build_pca_features(
    df: pd.DataFrame,
    macro_vars: list[str],
    prefix: str,
    max_components: int = 2,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None]:
    available = [var for var in macro_vars if var in df.columns]
    if len(available) < 2:
        return df.copy(), [], None

    macro_complete = df[available].dropna().copy()
    if macro_complete.shape[0] < 3:
        return df.copy(), [], None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(macro_complete)
    n_components = min(max_components, len(available), macro_complete.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(scaled)

    out = df.copy()
    pc_names = []
    for i in range(n_components):
        col = f"{prefix}_pc{i + 1}"
        out.loc[macro_complete.index, col] = pcs[:, i]
        pc_names.append(col)

    explained = pd.DataFrame(
        {
            "component": pc_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return out, pc_names, explained


df = pd.read_csv(DATA_PATH)
df["start_year"] = df["year_label"].str[:4].astype(int)
df = df.sort_values("start_year").reset_index(drop=True)

# Growth variables that can be used in both workflows.
df["investment_growth"] = make_growth(df["total_investment_rs_cr"])
df["goods_earnings_growth"] = make_growth(df["goods_earnings_rs_cr"])
df["freight_growth"] = make_growth(df["total_revenue_traffic_tonnes_originating"])
df["passenger_km_growth"] = make_growth(df["passenger_kilometres_million"])
df["working_expenditure_growth"] = make_growth(df["total_working_expenditure_in_cr"])

# Lagged variables for the forecasting model.
lag_map = {
    "lag_rail_gdp_growth": "real_gdp_growth_of_railways",
    "lag_investment_growth": "investment_growth",
    "lag_goods_earnings_growth": "goods_earnings_growth",
    "lag_freight_growth": "freight_growth",
    "lag_passenger_km_growth": "passenger_km_growth",
    "lag_operating_ratio": "operating_ratio",
    "lag_fiscal_deficit": "fiscal_deficit_gdp",
    "lag_gov_expenditure": "gov_expenditure_gdp_2024",
    "lag_real_interest_rate": "real_interest_rate",
    "lag_gdp_deflator": "gdp_deflator_growth_rate_y_y_chg",
    "lag_energy_inflation": "energy_price_inflation_y_y_chg",
}
for lag_name, source in lag_map.items():
    df[lag_name] = df[source].shift(1)


# ---------------------------
# Model 1: Current variables -> current-year 2024-25 nowcast
# ---------------------------
current_railway_candidates = [
    "investment_growth",
    "goods_earnings_growth",
    "freight_growth",
    "operating_ratio",
]
current_macro_candidates = [
    "fiscal_deficit_gdp",
    "gov_expenditure_gdp_2024",
    "gdp_deflator_growth_rate_y_y_chg",
]
current_candidates = current_railway_candidates + current_macro_candidates

current_selection_data = df.loc[df["real_gdp_growth_of_railways"].notna(), ["year_label", "start_year", "real_gdp_growth_of_railways"] + current_candidates].dropna().copy()
current_selection, current_selected, current_lasso_pipe = run_lasso_selection(current_selection_data, "real_gdp_growth_of_railways", current_candidates)
current_selection.to_csv(OUTPUT_DIR / "prof_current_lasso_selection.csv", index=False)

current_selected_railway = [v for v in current_selected if v in current_railway_candidates]
current_selected_macro = [v for v in current_selected if v in current_macro_candidates]

current_data_with_pca, current_pc_names, current_pca_summary = build_pca_features(
    current_selection_data[["year_label", "start_year", "real_gdp_growth_of_railways"] + current_selected].copy(),
    current_selected_macro,
    prefix="current_macro",
)
if current_pca_summary is not None:
    current_pca_summary.to_csv(OUTPUT_DIR / "prof_current_pca_summary.csv", index=False)

current_macro_features = current_pc_names if current_pc_names else current_selected_macro
current_final_predictors = current_selected_railway + current_macro_features
if not current_final_predictors:
    current_final_predictors = current_selected[:]

current_final_model_data = current_data_with_pca[["year_label", "start_year", "real_gdp_growth_of_railways"] + current_final_predictors].dropna().copy()
current_model = LinearRegression()
current_model.fit(current_final_model_data[current_final_predictors], current_final_model_data["real_gdp_growth_of_railways"])

current_full_pred = current_model.predict(current_final_model_data[current_final_predictors])
current_summary = pd.DataFrame(
    [
        {
            "workflow": "current_variables_then_pca_then_ols",
            "lasso_selected": ", ".join(current_selected),
            "final_predictors_used": ", ".join(current_final_predictors),
            "observations": len(current_final_model_data),
            "r_squared": float(current_model.score(current_final_model_data[current_final_predictors], current_final_model_data["real_gdp_growth_of_railways"])),
            "rmse": rmse(current_final_model_data["real_gdp_growth_of_railways"].to_numpy(), current_full_pred),
        }
    ]
)
current_summary.to_csv(OUTPUT_DIR / "prof_current_model_summary.csv", index=False)

current_coefficients = pd.DataFrame(
    [{"term": "intercept", "estimate": float(current_model.intercept_)}]
    + [{"term": var, "estimate": float(coef)} for var, coef in zip(current_final_predictors, current_model.coef_)]
)
current_coefficients.to_csv(OUTPUT_DIR / "prof_current_model_coefficients.csv", index=False)

# Predict 2024-25 with current-year available variables.
current_2024_base = df.loc[df["year_label"] == "2024-25", ["year_label", "start_year"] + current_selected].copy()
current_2024_with_pca = current_2024_base.copy()
if current_selected_macro and current_pc_names:
    train_macro = current_selection_data[current_selected_macro].dropna().copy()
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_macro)
    pca = PCA(n_components=len(current_pc_names), random_state=42)
    pca.fit(scaled_train)
    macro_row = current_2024_with_pca[current_selected_macro]
    scaled_row = scaler.transform(macro_row)
    pcs = pca.transform(scaled_row)
    for idx, col in enumerate(current_pc_names):
        current_2024_with_pca[col] = pcs[:, idx]

current_2024_prediction = float(current_model.predict(current_2024_with_pca[current_final_predictors])[0])
pd.DataFrame(
    [
        {
            "forecast_year": "2024-25",
            "prediction_type": "current_year_nowcast",
            "predicted_real_gdp_growth_of_railways": current_2024_prediction,
        }
    ]
).to_csv(OUTPUT_DIR / "prof_current_2024_25_prediction.csv", index=False)


# ---------------------------
# Model 2: Lagged variables -> 5-year and 10-year forecasts
# ---------------------------
lagged_railway_candidates = [
    "lag_rail_gdp_growth",
    "lag_investment_growth",
    "lag_goods_earnings_growth",
    "lag_freight_growth",
    "lag_operating_ratio",
]
lagged_macro_candidates = [
    "lag_fiscal_deficit",
    "lag_gov_expenditure",
    "lag_gdp_deflator",
]
lagged_candidates = lagged_railway_candidates + lagged_macro_candidates

lagged_selection_data = df.loc[df["real_gdp_growth_of_railways"].notna(), ["year_label", "start_year", "real_gdp_growth_of_railways"] + lagged_candidates].dropna().copy()
lagged_selection, lagged_selected, lagged_lasso_pipe = run_lasso_selection(lagged_selection_data, "real_gdp_growth_of_railways", lagged_candidates)
lagged_selection.to_csv(OUTPUT_DIR / "prof_lagged_lasso_selection.csv", index=False)

lagged_selected_railway = [v for v in lagged_selected if v in lagged_railway_candidates]
lagged_selected_macro = [v for v in lagged_selected if v in lagged_macro_candidates]

lagged_data_with_pca, lagged_pc_names, lagged_pca_summary = build_pca_features(
    lagged_selection_data[["year_label", "start_year", "real_gdp_growth_of_railways"] + lagged_selected].copy(),
    lagged_selected_macro,
    prefix="lagged_macro",
)
if lagged_pca_summary is not None:
    lagged_pca_summary.to_csv(OUTPUT_DIR / "prof_lagged_pca_summary.csv", index=False)

lagged_macro_features = lagged_pc_names if lagged_pc_names else lagged_selected_macro
lagged_final_predictors = lagged_selected_railway + lagged_macro_features
if not lagged_final_predictors:
    lagged_final_predictors = lagged_selected[:]

lagged_model_data = lagged_data_with_pca[["year_label", "start_year", "real_gdp_growth_of_railways"] + lagged_final_predictors].dropna().copy()
lagged_train = lagged_model_data.iloc[:-3].copy()
lagged_test = lagged_model_data.iloc[-3:].copy()

lagged_model = LinearRegression()
lagged_model.fit(lagged_train[lagged_final_predictors], lagged_train["real_gdp_growth_of_railways"])
lagged_train_pred = lagged_model.predict(lagged_train[lagged_final_predictors])
lagged_test_pred = lagged_model.predict(lagged_test[lagged_final_predictors])

lagged_summary = pd.DataFrame(
    [
        {
            "workflow": "lagged_variables_then_pca_then_ols",
            "lasso_selected": ", ".join(lagged_selected),
            "final_predictors_used": ", ".join(lagged_final_predictors),
            "train_observations": len(lagged_train),
            "test_observations": len(lagged_test),
            "train_rmse": rmse(lagged_train["real_gdp_growth_of_railways"].to_numpy(), lagged_train_pred),
            "test_rmse": rmse(lagged_test["real_gdp_growth_of_railways"].to_numpy(), lagged_test_pred),
            "test_mae": mae(lagged_test["real_gdp_growth_of_railways"].to_numpy(), lagged_test_pred),
        }
    ]
)
lagged_summary.to_csv(OUTPUT_DIR / "prof_lagged_model_summary.csv", index=False)

lagged_coefficients = pd.DataFrame(
    [{"term": "intercept", "estimate": float(lagged_model.intercept_)}]
    + [{"term": var, "estimate": float(coef)} for var, coef in zip(lagged_final_predictors, lagged_model.coef_)]
)
lagged_coefficients.to_csv(OUTPUT_DIR / "prof_lagged_model_coefficients.csv", index=False)

# Build the first lagged forecast for 2024-25 using actual lagged inputs.
lagged_2024_base = df.loc[df["year_label"] == "2024-25", ["year_label", "start_year"] + lagged_selected].copy()
lagged_2024_with_pca = lagged_2024_base.copy()
if lagged_selected_macro and lagged_pc_names:
    train_macro = lagged_selection_data[lagged_selected_macro].dropna().copy()
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_macro)
    pca = PCA(n_components=len(lagged_pc_names), random_state=42)
    pca.fit(scaled_train)
    macro_row = lagged_2024_with_pca[lagged_selected_macro]
    scaled_row = scaler.transform(macro_row)
    pcs = pca.transform(scaled_row)
    for idx, col in enumerate(lagged_pc_names):
        lagged_2024_with_pca[col] = pcs[:, idx]

first_lagged_forecast = float(lagged_model.predict(lagged_2024_with_pca[lagged_final_predictors])[0])

# Build assumptions for multi-step forecasts.
exogenous_assumptions = {}
for var in lagged_selected_railway:
    if var == "lag_rail_gdp_growth":
        continue
    base_var = var.replace("lag_", "")
    recent = df[base_var].dropna().tail(3)
    exogenous_assumptions[var] = {
        "baseline": float(recent.mean()),
        "optimistic": float(recent.mean() + 0.5 * recent.std(ddof=0)),
        "pessimistic": float(recent.mean() - 0.5 * recent.std(ddof=0)),
    }
for pc in lagged_pc_names:
    recent = lagged_data_with_pca[pc].dropna().tail(3)
    exogenous_assumptions[pc] = {
        "baseline": float(recent.mean()),
        "optimistic": float(recent.mean() + 0.5 * recent.std(ddof=0)),
        "pessimistic": float(recent.mean() - 0.5 * recent.std(ddof=0)),
    }

scenario_rows = []
for scenario in ["baseline", "optimistic", "pessimistic"]:
    prev_growth = first_lagged_forecast
    for horizon in range(1, 11):
        forecast_start_year = 2023 + horizon
        if horizon == 1:
            predicted_growth = first_lagged_forecast
        else:
            x = {}
            for variable in lagged_final_predictors:
                if variable == "lag_rail_gdp_growth":
                    x[variable] = prev_growth
                else:
                    x[variable] = exogenous_assumptions[variable][scenario]
            predicted_growth = float(lagged_model.predict(pd.DataFrame([x]))[0])
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
scenario_df.to_csv(OUTPUT_DIR / "prof_lagged_scenario_forecast_path.csv", index=False)

baseline_path = scenario_df.loc[scenario_df["scenario"] == "baseline"].reset_index(drop=True)
horizon_df = pd.DataFrame(
    [
        {
            "horizon": "short_term_1_year",
            "forecast_year": baseline_path.loc[0, "forecast_year"],
            "predicted_real_gdp_growth_of_railways": baseline_path.loc[0, "predicted_real_gdp_growth_of_railways"],
        },
        {
            "horizon": "medium_term_5_year",
            "forecast_year": baseline_path.loc[4, "forecast_year"],
            "predicted_real_gdp_growth_of_railways": baseline_path.loc[4, "predicted_real_gdp_growth_of_railways"],
        },
        {
            "horizon": "long_run_10_year",
            "forecast_year": baseline_path.loc[9, "forecast_year"],
            "predicted_real_gdp_growth_of_railways": baseline_path.loc[9, "predicted_real_gdp_growth_of_railways"],
        },
    ]
)
horizon_df.to_csv(OUTPUT_DIR / "prof_lagged_horizon_forecasts.csv", index=False)


report_lines = [
    "# Professor-Requested Workflow",
    "",
    "## Structure",
    "",
    "1. Lasso for variable selection",
    "2. PCA on the selected macro block",
    "3. Current-variable model for 2024-25 nowcasting",
    "4. Lagged-variable model for medium-run and long-run forecasting",
    "",
    "## Current-Variable Model",
    "",
    f"- Lasso selected: {', '.join(current_selected)}",
    f"- Final predictors after PCA step: {', '.join(current_final_predictors)}",
    f"- Predicted 2024-25 railways GDP growth: {current_2024_prediction:.2f}%",
    f"- PCA note: {'PCA was applied to the selected macro block.' if current_pc_names else 'After Lasso, there were not enough selected macro variables to form principal components, so the selected macro variable was kept directly.'}",
    "",
    "## Lagged Forecasting Model",
    "",
    f"- Lasso selected: {', '.join(lagged_selected)}",
    f"- Final predictors after PCA step: {', '.join(lagged_final_predictors)}",
    f"- 1-year forecast: {horizon_df.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 5-year forecast: {horizon_df.loc[1, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 10-year forecast: {horizon_df.loc[2, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- PCA note: {'PCA was applied to the selected lagged macro block.' if lagged_pc_names else 'After Lasso, there were not enough selected lagged macro variables to form principal components, so PCA did not add extra components here.'}",
    "",
    "## Interpretation",
    "",
    "The current-variable model is best described as a nowcasting or contemporaneous prediction model because it uses variables from the same year to explain 2024-25 railways GDP growth. The lagged-variable model is the true forecasting model because it uses only past information.",
]
(DOCS_DIR / "professor_requested_workflow.md").write_text("\n".join(report_lines))

print("Current selected:", ", ".join(current_selected))
print("Lagged selected:", ", ".join(lagged_selected))
