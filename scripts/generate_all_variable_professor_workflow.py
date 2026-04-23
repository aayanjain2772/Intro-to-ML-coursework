from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def run_lasso_allvars(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, list[str], Pipeline]:
    cv_folds = min(5, len(X))
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=cv_folds, random_state=42, max_iter=200000)),
        ]
    )
    pipe.fit(X, y)
    lasso = pipe.named_steps["lasso"]
    selection = pd.DataFrame(
        {
            "variable": X.columns,
            "lasso_coefficient": lasso.coef_,
            "selected": ["yes" if abs(coef) > 1e-8 else "no" for coef in lasso.coef_],
        }
    )
    selected = selection.loc[selection["selected"] == "yes", "variable"].tolist()
    if not selected:
        # Keep the strongest absolute coefficient if Lasso shrinks everything.
        strongest_idx = int(np.argmax(np.abs(lasso.coef_)))
        selected = [X.columns[strongest_idx]]
        selection.loc[selection["variable"] == selected[0], "selected"] = "yes"
    return selection, selected, pipe


def fit_pca_model(X: pd.DataFrame, y: pd.Series, variance_cutoff: float = 0.9):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    if X.shape[1] >= 2:
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_cutoff) + 1)
        n_components = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        feature_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
        pca_summary = pd.DataFrame(
            {
                "component": feature_names,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            }
        )
    else:
        pca = None
        X_pca = X_scaled
        feature_names = list(X.columns)
        pca_summary = pd.DataFrame(
            {
                "component": feature_names,
                "explained_variance_ratio": [1.0],
                "cumulative_explained_variance": [1.0],
            }
        )

    model = LinearRegression()
    model.fit(X_pca, y)
    return {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names,
        "pca_summary": pca_summary,
        "model": model,
        "fitted": model.predict(X_pca),
    }


def transform_with_pca(fit_obj, X: pd.DataFrame) -> np.ndarray:
    X_imputed = fit_obj["imputer"].transform(X)
    X_scaled = fit_obj["scaler"].transform(X_imputed)
    if fit_obj["pca"] is not None:
        return fit_obj["pca"].transform(X_scaled)
    return X_scaled


df = pd.read_csv(DATA_PATH)
df["start_year"] = df["year_label"].str[:4].astype(int)
df = df.sort_values("start_year").reset_index(drop=True)

target = "real_gdp_growth_of_railways"
numeric_cols = [c for c in df.columns if c != "year_label" and pd.api.types.is_numeric_dtype(df[c])]
base_predictors = [c for c in numeric_cols if c not in [target, "start_year"]]

# Current-variable branch: all current numeric variables in the dataset except the target.
current_train_df = df.loc[df[target].notna(), ["year_label", "start_year", target] + base_predictors].copy()
X_current = current_train_df[base_predictors]
y_current = current_train_df[target]
current_selection, current_selected, current_lasso_pipe = run_lasso_allvars(X_current, y_current)
current_selection.to_csv(OUTPUT_DIR / "allvars_current_lasso_selection.csv", index=False)

current_selected_df = current_train_df[current_selected]
current_fit = fit_pca_model(current_selected_df, y_current)
current_fit["pca_summary"].to_csv(OUTPUT_DIR / "allvars_current_pca_summary.csv", index=False)

current_pred_train = current_fit["fitted"]
pd.DataFrame(
    [
        {
            "workflow": "all_variables_current_then_lasso_then_pca_then_ols",
            "candidate_variable_count": len(base_predictors),
            "lasso_selected_count": len(current_selected),
            "lasso_selected_variables": ", ".join(current_selected),
            "pca_feature_count": len(current_fit["feature_names"]),
            "r_squared": float(current_fit["model"].score(transform_with_pca(current_fit, current_selected_df), y_current)),
            "rmse": rmse(y_current.to_numpy(), current_pred_train),
        }
    ]
).to_csv(OUTPUT_DIR / "allvars_current_model_summary.csv", index=False)

current_coef_df = pd.DataFrame(
    [{"term": "intercept", "estimate": float(current_fit["model"].intercept_)}]
    + [{"term": name, "estimate": float(coef)} for name, coef in zip(current_fit["feature_names"], current_fit["model"].coef_)]
)
current_coef_df.to_csv(OUTPUT_DIR / "allvars_current_model_coefficients.csv", index=False)

current_2024_X = df.loc[df["year_label"] == "2024-25", current_selected]
current_2024_pred = float(current_fit["model"].predict(transform_with_pca(current_fit, current_2024_X))[0])
pd.DataFrame(
    [
        {
            "forecast_year": "2024-25",
            "prediction_type": "current_year_nowcast_all_variables",
            "predicted_real_gdp_growth_of_railways": current_2024_pred,
        }
    ]
).to_csv(OUTPUT_DIR / "allvars_current_2024_25_prediction.csv", index=False)


# Lagged-variable branch: lag every numeric variable including the target.
lagged_df = df[["year_label", "start_year", target] + base_predictors].copy()
lag_source_cols = [target] + base_predictors
for col in lag_source_cols:
    lagged_df[f"lag_{col}"] = lagged_df[col].shift(1)

lag_predictors = [f"lag_{col}" for col in lag_source_cols]
lag_train_df = lagged_df.loc[lagged_df[target].notna(), ["year_label", "start_year", target] + lag_predictors].copy()
X_lag = lag_train_df[lag_predictors]
y_lag = lag_train_df[target]
lag_selection, lag_selected, lag_lasso_pipe = run_lasso_allvars(X_lag, y_lag)
lag_selection.to_csv(OUTPUT_DIR / "allvars_lagged_lasso_selection.csv", index=False)

lag_selected_train_df = lag_train_df[lag_selected].copy()

# Holdout split for lagged forecasting branch.
lag_model_data = lag_train_df[["year_label", "start_year", target] + lag_selected].copy()
lag_train = lag_model_data.iloc[:-3].copy()
lag_test = lag_model_data.iloc[-3:].copy()
lag_fit = fit_pca_model(lag_train[lag_selected], lag_train[target])
lag_fit["pca_summary"].to_csv(OUTPUT_DIR / "allvars_lagged_pca_summary.csv", index=False)

lag_train_pred = lag_fit["model"].predict(transform_with_pca(lag_fit, lag_train[lag_selected]))
lag_test_pred = lag_fit["model"].predict(transform_with_pca(lag_fit, lag_test[lag_selected]))
pd.DataFrame(
    [
        {
            "workflow": "all_variables_lagged_then_lasso_then_pca_then_ols",
            "candidate_variable_count": len(lag_predictors),
            "lasso_selected_count": len(lag_selected),
            "lasso_selected_variables": ", ".join(lag_selected),
            "pca_feature_count": len(lag_fit["feature_names"]),
            "train_observations": len(lag_train),
            "test_observations": len(lag_test),
            "train_rmse": rmse(lag_train[target].to_numpy(), lag_train_pred),
            "test_rmse": rmse(lag_test[target].to_numpy(), lag_test_pred),
            "test_mae": mae(lag_test[target].to_numpy(), lag_test_pred),
        }
    ]
).to_csv(OUTPUT_DIR / "allvars_lagged_model_summary.csv", index=False)

lag_coef_df = pd.DataFrame(
    [{"term": "intercept", "estimate": float(lag_fit["model"].intercept_)}]
    + [{"term": name, "estimate": float(coef)} for name, coef in zip(lag_fit["feature_names"], lag_fit["model"].coef_)]
)
lag_coef_df.to_csv(OUTPUT_DIR / "allvars_lagged_model_coefficients.csv", index=False)

# First forecast using actual 2024-25 lagged values.
first_forecast_X = lagged_df.loc[lagged_df["year_label"] == "2024-25", lag_selected]
first_lagged_forecast = float(lag_fit["model"].predict(transform_with_pca(lag_fit, first_forecast_X))[0])
pd.DataFrame(
    [
        {
            "forecast_year": "2024-25",
            "prediction_type": "lagged_forecast_all_variables",
            "predicted_real_gdp_growth_of_railways": first_lagged_forecast,
        }
    ]
).to_csv(OUTPUT_DIR / "allvars_lagged_2024_25_prediction.csv", index=False)

# Build recursive forecast scenarios.
selected_exogenous = [var for var in lag_selected if var != f"lag_{target}"]
assumptions = {}
for var in selected_exogenous:
    source_col = var.replace("lag_", "", 1)
    recent = df[source_col].dropna().tail(3)
    baseline = float(recent.mean())
    sd = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
    assumptions[var] = {
        "baseline": baseline,
        "optimistic": baseline + 0.5 * sd,
        "pessimistic": baseline - 0.5 * sd,
    }

scenario_rows = []
for scenario in ["baseline", "optimistic", "pessimistic"]:
    prev_growth = first_lagged_forecast
    for horizon in range(1, 11):
        forecast_start_year = 2023 + horizon
        if horizon == 1:
            predicted = first_lagged_forecast
        else:
            row = {}
            for var in lag_selected:
                if var == f"lag_{target}":
                    row[var] = prev_growth
                else:
                    row[var] = assumptions[var][scenario]
            predicted = float(lag_fit["model"].predict(transform_with_pca(lag_fit, pd.DataFrame([row])))[0])
        scenario_rows.append(
            {
                "scenario": scenario,
                "horizon_years_ahead": horizon,
                "forecast_year": next_year_label(forecast_start_year),
                "predicted_real_gdp_growth_of_railways": predicted,
            }
        )
        prev_growth = predicted

scenario_df = pd.DataFrame(scenario_rows)
scenario_df.to_csv(OUTPUT_DIR / "allvars_lagged_scenario_forecast_path.csv", index=False)

baseline_path = scenario_df.loc[scenario_df["scenario"] == "baseline"].reset_index(drop=True)
pd.DataFrame(
    [
        {"horizon": "short_term_1_year", "forecast_year": baseline_path.loc[0, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[0, "predicted_real_gdp_growth_of_railways"]},
        {"horizon": "medium_term_5_year", "forecast_year": baseline_path.loc[4, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[4, "predicted_real_gdp_growth_of_railways"]},
        {"horizon": "long_run_10_year", "forecast_year": baseline_path.loc[9, "forecast_year"], "predicted_real_gdp_growth_of_railways": baseline_path.loc[9, "predicted_real_gdp_growth_of_railways"]},
    ]
).to_csv(OUTPUT_DIR / "allvars_lagged_horizon_forecasts.csv", index=False)


report_lines = [
    "# All-Variable Professor Workflow",
    "",
    "This workflow lets Lasso see the full cleaned numeric dataset rather than a hand-curated candidate subset.",
    "",
    "## Current-Variable Branch",
    "",
    f"- Candidate variables given to Lasso: {len(base_predictors)}",
    f"- Lasso selected count: {len(current_selected)}",
    f"- Lasso selected variables: {', '.join(current_selected)}",
    f"- PCA features used in final model: {', '.join(current_fit['feature_names'])}",
    f"- 2024-25 current-year nowcast: {current_2024_pred:.2f}%",
    "",
    "## Lagged Branch",
    "",
    f"- Candidate lagged variables given to Lasso: {len(lag_predictors)}",
    f"- Lasso selected count: {len(lag_selected)}",
    f"- Lasso selected variables: {', '.join(lag_selected)}",
    f"- PCA features used in final lagged model: {', '.join(lag_fit['feature_names'])}",
    f"- 2024-25 lagged forecast: {baseline_path.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 5-year lagged forecast: {baseline_path.loc[4, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- 10-year lagged forecast: {baseline_path.loc[9, 'predicted_real_gdp_growth_of_railways']:.2f}%",
]
(DOCS_DIR / "allvars_professor_workflow.md").write_text("\n".join(report_lines))

print("Current selected variables:", ", ".join(current_selected))
print("Lagged selected variables:", ", ".join(lag_selected))
