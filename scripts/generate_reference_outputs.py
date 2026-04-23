from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from statistics import mean, median, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "railways_master.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def read_rows() -> list[dict]:
    with DATA_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {}
            for key, value in row.items():
                if value == "":
                    clean[key] = None
                    continue
                if key == "year_label":
                    clean[key] = value
                    clean["start_year"] = int(value[:4])
                    continue
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            rows.append(clean)
    return sorted(rows, key=lambda x: x["start_year"])


def pct_change(values: list[float | None]) -> list[float | None]:
    out = [None]
    for prev, curr in zip(values, values[1:]):
        if prev in (None, 0) or curr is None:
            out.append(None)
        else:
            out.append((curr - prev) / prev * 100.0)
    return out


def lag(values: list[float | None]) -> list[float | None]:
    return [None] + values[:-1]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = []
    for row in a:
        out_row = []
        for col in zip(*b):
            out_row.append(sum(x * y for x, y in zip(row, col)))
        out.append(out_row)
    return out


def invert(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col] = [v / pivot_val for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [v - factor * base for v, base in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def ols_fit(data: list[dict], target: str, predictors: list[str]) -> dict:
    X = []
    y = []
    used = []
    for row in data:
        values = [row.get(target)] + [row.get(p) for p in predictors]
        if any(v is None for v in values):
            continue
        used.append(row)
        X.append([1.0] + [float(row[p]) for p in predictors])
        y.append([float(row[target])])

    xt = transpose(X)
    xtx = matmul(xt, X)
    xtx_inv = invert(xtx)
    xty = matmul(xt, y)
    beta_col = matmul(xtx_inv, xty)
    beta = [b[0] for b in beta_col]

    fitted = []
    residuals = []
    for row_vec, y_val in zip(X, [yy[0] for yy in y]):
        pred = sum(b * x for b, x in zip(beta, row_vec))
        fitted.append(pred)
        residuals.append(y_val - pred)

    n = len(y)
    k = len(predictors)
    y_vals = [yy[0] for yy in y]
    y_bar = mean(y_vals)
    rss = sum(res ** 2 for res in residuals)
    tss = sum((val - y_bar) ** 2 for val in y_vals)
    r2 = 1.0 - rss / tss if tss else 0.0
    adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else float("nan")
    sigma2 = rss / (n - k - 1) if n - k - 1 > 0 else float("nan")
    aic = n * math.log(rss / n) + 2 * (k + 1) if n and rss > 0 else float("nan")

    return {
        "data": used,
        "predictors": predictors,
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
        "sigma": math.sqrt(sigma2) if sigma2 == sigma2 and sigma2 >= 0 else float("nan"),
        "aic": aic,
    }


def predict_row(row: dict, beta: list[float], predictors: list[str]) -> float:
    values = [1.0] + [float(row[p]) for p in predictors]
    return sum(b * x for b, x in zip(beta, values))


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def series_svg(path: Path, years: list[int], series: list[tuple[str, list[float], str]], title: str, y_label: str) -> None:
    width, height = 900, 420
    margin = {"left": 70, "right": 30, "top": 50, "bottom": 50}
    xs = years
    ys = [v for _, vals, _ in series for v in vals if v is not None]
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
      min_y -= 1.0
      max_y += 1.0
    def x_map(year: int) -> float:
        return margin["left"] + (year - min(xs)) / (max(xs) - min(xs)) * (width - margin["left"] - margin["right"])
    def y_map(value: float) -> float:
        return height - margin["bottom"] - (value - min_y) / (max_y - min_y) * (height - margin["top"] - margin["bottom"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
        f'<line x1="{margin["left"]}" y1="{height-margin["bottom"]}" x2="{width-margin["right"]}" y2="{height-margin["bottom"]}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height-margin["bottom"]}" stroke="#333"/>',
        f'<text x="22" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90,22,{height/2})" font-family="Arial">{y_label}</text>',
    ]

    for year in xs:
        x = x_map(year)
        parts.append(f'<line x1="{x}" y1="{height-margin["bottom"]}" x2="{x}" y2="{height-margin["bottom"]+6}" stroke="#333"/>')
        parts.append(f'<text x="{x}" y="{height-margin["bottom"]+20}" text-anchor="middle" font-size="11" font-family="Arial">{year}</text>')

    for tick in range(6):
        val = min_y + (max_y - min_y) * tick / 5
        y = y_map(val)
        parts.append(f'<line x1="{margin["left"]-5}" y1="{y}" x2="{margin["left"]}" y2="{y}" stroke="#333"/>')
        parts.append(f'<text x="{margin["left"]-10}" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial">{val:.1f}</text>')
        parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width-margin["right"]}" y2="{y}" stroke="#e6e6e6"/>')

    legend_x = width - 200
    legend_y = 60
    for idx, (label, vals, color) in enumerate(series):
        points = " ".join(f"{x_map(year)},{y_map(val)}" for year, val in zip(xs, vals) if val is not None)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for year, val in zip(xs, vals):
            if val is not None:
                parts.append(f'<circle cx="{x_map(year)}" cy="{y_map(val)}" r="3.5" fill="{color}"/>')
        ly = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+22}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x+30}" y="{ly+4}" font-size="12" font-family="Arial">{label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


rows = read_rows()

for key, source in [
    ("investment_growth", "total_investment_rs_cr"),
    ("goods_earnings_growth", "goods_earnings_rs_cr"),
    ("freight_growth", "total_revenue_traffic_tonnes_originating"),
    ("passenger_km_growth", "passenger_kilometres_million"),
]:
    growths = pct_change([row[source] for row in rows])
    for row, value in zip(rows, growths):
        row[key] = value

for key, source in [
    ("lag_rail_gdp_growth", "real_gdp_growth_of_railways"),
    ("lag_investment_growth", "investment_growth"),
    ("lag_goods_earnings_growth", "goods_earnings_growth"),
    ("lag_freight_growth", "freight_growth"),
    ("lag_passenger_km_growth", "passenger_km_growth"),
    ("lag_fiscal_deficit", "fiscal_deficit_gdp"),
    ("lag_real_interest_rate", "real_interest_rate"),
    ("lag_gdp_deflator", "gdp_deflator_growth_rate_y_y_chg"),
]:
    lags = lag([row[source] for row in rows])
    for row, value in zip(rows, lags):
        row[key] = value

eda_vars = [
    "real_gdp_growth_of_railways",
    "investment_growth",
    "goods_earnings_growth",
    "freight_growth",
    "lag_rail_gdp_growth",
    "lag_investment_growth",
    "lag_goods_earnings_growth",
    "lag_freight_growth",
    "lag_fiscal_deficit",
    "lag_gdp_deflator",
]

eda_rows = []
for var in eda_vars:
    vals = [row[var] for row in rows if row[var] is not None]
    eda_rows.append(
        {
            "variable": var,
            "non_missing": len(vals),
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "sd": round(stdev(vals), 4) if len(vals) > 1 else "",
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    )
write_csv(OUTPUT_DIR / "eda_summary.csv", list(eda_rows[0].keys()), eda_rows)

missing_rows = [{"variable": key, "missing_count": sum(1 for row in rows if row.get(key) is None)} for key in rows[0].keys()]
write_csv(OUTPUT_DIR / "missingness_summary.csv", ["variable", "missing_count"], missing_rows)

candidate_specs = {
    "baseline": ["lag_rail_gdp_growth"],
    "freight_investment": ["lag_rail_gdp_growth", "lag_freight_growth", "lag_investment_growth"],
    "commercial_investment": ["lag_rail_gdp_growth", "lag_goods_earnings_growth", "lag_investment_growth"],
    "demand_mix": ["lag_rail_gdp_growth", "lag_goods_earnings_growth", "lag_freight_growth"],
}

comparison = []
for name, predictors in candidate_specs.items():
    usable = [row for row in rows if row["real_gdp_growth_of_railways"] is not None and all(row.get(p) is not None for p in predictors)]
    train = usable[:-3]
    test = usable[-3:]
    fit = ols_fit(train, "real_gdp_growth_of_railways", predictors)
    train_pred = [predict_row(row, fit["beta"], predictors) for row in train]
    test_pred = [predict_row(row, fit["beta"], predictors) for row in test]
    comparison.append(
        {
            "model": name,
            "observations": len(usable),
            "train_observations": len(train),
            "test_observations": len(test),
            "train_rmse": round(rmse([r["real_gdp_growth_of_railways"] for r in train], train_pred), 4),
            "test_rmse": round(rmse([r["real_gdp_growth_of_railways"] for r in test], test_pred), 4),
            "test_mae": round(mae([r["real_gdp_growth_of_railways"] for r in test], test_pred), 4),
            "adjusted_r_squared": round(fit["adj_r2"], 4),
            "aic": round(fit["aic"], 4),
        }
    )

comparison.sort(key=lambda r: r["test_rmse"])
write_csv(OUTPUT_DIR / "model_comparison.csv", list(comparison[0].keys()), comparison)

best_model_name = comparison[0]["model"]
best_predictors = candidate_specs[best_model_name]
full_fit = ols_fit(rows, "real_gdp_growth_of_railways", best_predictors)

coef_rows = []
for term, estimate in zip(["intercept"] + best_predictors, full_fit["beta"]):
    coef_rows.append({"term": term, "estimate": round(estimate, 6)})
write_csv(OUTPUT_DIR / "final_model_coefficients.csv", ["term", "estimate"], coef_rows)

prediction_rows = []
for row, fitted, residual in zip(full_fit["data"], full_fit["fitted"], full_fit["residuals"]):
    prediction_rows.append(
        {
            "year_label": row["year_label"],
            "actual": round(row["real_gdp_growth_of_railways"], 6),
            "fitted": round(fitted, 6),
            "residual": round(residual, 6),
        }
    )
write_csv(OUTPUT_DIR / "predictions.csv", ["year_label", "actual", "fitted", "residual"], prediction_rows)

actual_vals = [r["actual"] for r in prediction_rows]
fitted_vals = [r["fitted"] for r in prediction_rows]
metric_rows = [
    {
        "model": best_model_name,
        "observations": len(prediction_rows),
        "r_squared": round(full_fit["r2"], 4),
        "adjusted_r_squared": round(full_fit["adj_r2"], 4),
        "residual_standard_error": round(full_fit["sigma"], 4),
        "in_sample_rmse": round(rmse(actual_vals, fitted_vals), 4),
        "in_sample_mae": round(mae(actual_vals, fitted_vals), 4),
    }
]
write_csv(OUTPUT_DIR / "final_model_metrics.csv", list(metric_rows[0].keys()), metric_rows)

all_fields = sorted({key for row in full_fit["data"] for key in row.keys()})
write_csv(OUTPUT_DIR / "model_dataset.csv", all_fields, full_fit["data"])

forecast_rows = [row for row in rows if row["year_label"] == "2024-25" and all(row.get(p) is not None for p in best_predictors)]
forecast_text = "A 2024-25 forecast was not generated because at least one lagged predictor was unavailable."
if forecast_rows:
    forecast_value = predict_row(forecast_rows[0], full_fit["beta"], best_predictors)
    write_csv(
        OUTPUT_DIR / "forecast_2024_25.csv",
        ["forecast_year", "predicted_real_gdp_growth_of_railways"],
        [{"forecast_year": "2024-25", "predicted_real_gdp_growth_of_railways": round(forecast_value, 6)}],
    )
    forecast_text = f"Using the selected lagged linear regression, the predicted railways real GDP growth for 2024-25 is {forecast_value:.2f}%."

years = [row["start_year"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
target_vals = [row["real_gdp_growth_of_railways"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
freight_vals = [row["freight_growth"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
invest_vals = [row["investment_growth"] for row in rows if row["real_gdp_growth_of_railways"] is not None]

series_svg(
    OUTPUT_DIR / "eda_timeseries.svg",
    years,
    [
        ("Railways GDP growth", target_vals, "#1f4e79"),
        ("Freight growth", freight_vals, "#2e8b57"),
        ("Investment growth", invest_vals, "#b22222"),
    ],
    "Railways Growth and Demand Drivers",
    "Percent",
)

series_svg(
    OUTPUT_DIR / "actual_vs_predicted.svg",
    [row["start_year"] for row in full_fit["data"]],
    [
        ("Actual", [row["real_gdp_growth_of_railways"] for row in full_fit["data"]], "#1f4e79"),
        ("Fitted", full_fit["fitted"], "#b22222"),
    ],
    f"Actual vs Fitted ({best_model_name})",
    "Percent",
)

report_lines = [
    "# Railways Lagged Linear Regression",
    "",
    "## Objective",
    "",
    "Predict India railways sector real GDP growth using lagged linear regression features built from the provided annual railway and macro dataset.",
    "",
    "## What Was Done",
    "",
    "- Cleaned the master railway sheet into a usable annual CSV.",
    "- Created growth-rate features for investment, goods earnings, freight traffic, and passenger kilometres.",
    "- Created one-period lag variables so the regression only uses past information.",
    "- Compared four parsimonious lagged linear regression specifications on a small holdout window.",
    "",
    "## Selected Model",
    "",
    f"- Best holdout model: `{best_model_name}`",
    f"- Predictors: {', '.join(best_predictors)}",
    f"- Adjusted R-squared: {full_fit['adj_r2']:.3f}",
    f"- In-sample RMSE: {metric_rows[0]['in_sample_rmse']:.3f}",
    "",
    "## Forecast",
    "",
    forecast_text,
    "",
    "## Output Files",
    "",
    "- `outputs/eda_summary.csv`",
    "- `outputs/missingness_summary.csv`",
    "- `outputs/model_comparison.csv`",
    "- `outputs/final_model_coefficients.csv`",
    "- `outputs/final_model_metrics.csv`",
    "- `outputs/predictions.csv`",
    "- `outputs/eda_timeseries.svg`",
    "- `outputs/actual_vs_predicted.svg`",
]

(DOCS_DIR / "report.md").write_text("\n".join(report_lines))
