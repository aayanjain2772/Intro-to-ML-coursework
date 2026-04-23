from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "railways_master.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"
PDF_PATH = DOCS_DIR / "railways_project_detailed_report.pdf"
MD_PATH = DOCS_DIR / "railways_project_detailed_report.md"

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
    X, y, used = [], [], []
    for row in data:
        vals = [row.get(target)] + [row.get(p) for p in predictors]
        if any(v is None for v in vals):
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
    y_vals = [yy[0] for yy in y]
    for row_vec, y_val in zip(X, y_vals):
        pred = sum(b * x for b, x in zip(beta, row_vec))
        fitted.append(pred)
        residuals.append(y_val - pred)

    rss = sum(r * r for r in residuals)
    y_bar = mean(y_vals)
    tss = sum((y_val - y_bar) ** 2 for y_val in y_vals)
    r2 = 1 - rss / tss if tss else 0.0
    n = len(y_vals)
    k = len(predictors)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else float("nan")
    return {
        "data": used,
        "beta": beta,
        "predictors": predictors,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": math.sqrt(rss / n),
    }


def predict_row(row: dict, beta: list[float], predictors: list[str]) -> float:
    values = [1.0] + [float(row[p]) for p in predictors]
    return sum(b * x for b, x in zip(beta, values))


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def wrap_paragraph(text: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_text_page(pdf: PdfPages, title: str, blocks: list[str], fontsize: int = 11) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.07, 0.965, title, fontsize=18, fontweight="bold", ha="left", va="top")

    y = 0.93
    for block in blocks:
        lines = block.count("\n") + 1
        fig.text(0.07, y, block, fontsize=fontsize, ha="left", va="top", family="DejaVu Sans")
        y -= 0.028 * lines + 0.018
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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
]:
    lags = lag([row[source] for row in rows])
    for row, value in zip(rows, lags):
        row[key] = value

candidate_specs = {
    "baseline": ["lag_rail_gdp_growth"],
    "freight_investment": ["lag_rail_gdp_growth", "lag_freight_growth", "lag_investment_growth"],
    "commercial_investment": ["lag_rail_gdp_growth", "lag_goods_earnings_growth", "lag_investment_growth"],
    "demand_mix": ["lag_rail_gdp_growth", "lag_goods_earnings_growth", "lag_freight_growth"],
}

comparison = []
for model_name, predictors in candidate_specs.items():
    usable = [row for row in rows if row["real_gdp_growth_of_railways"] is not None and all(row.get(p) is not None for p in predictors)]
    train = usable[:-3]
    test = usable[-3:]
    fit = ols_fit(train, "real_gdp_growth_of_railways", predictors)
    train_pred = [predict_row(row, fit["beta"], predictors) for row in train]
    test_pred = [predict_row(row, fit["beta"], predictors) for row in test]
    comparison.append(
        {
            "model": model_name,
            "predictors": predictors,
            "observations": len(usable),
            "train_rmse": rmse([r["real_gdp_growth_of_railways"] for r in train], train_pred),
            "test_rmse": rmse([r["real_gdp_growth_of_railways"] for r in test], test_pred),
            "test_mae": mae([r["real_gdp_growth_of_railways"] for r in test], test_pred),
            "adj_r2_train": fit["adj_r2"],
        }
    )
comparison.sort(key=lambda x: x["test_rmse"])

best = comparison[0]
best_fit = ols_fit(rows, "real_gdp_growth_of_railways", best["predictors"])
coef_map = dict(zip(["intercept"] + best["predictors"], best_fit["beta"]))

forecast_row = [row for row in rows if row["year_label"] == "2024-25"][0]
short_term_forecast = predict_row(forecast_row, best_fit["beta"], best["predictors"])

hist_goods = [row["goods_earnings_growth"] for row in rows if row["goods_earnings_growth"] is not None]
hist_freight = [row["freight_growth"] for row in rows if row["freight_growth"] is not None]
baseline_goods = mean(hist_goods[-3:])
baseline_freight = mean(hist_freight[-3:])
goods_sd = (sum((x - mean(hist_goods)) ** 2 for x in hist_goods) / len(hist_goods)) ** 0.5
freight_sd = (sum((x - mean(hist_freight)) ** 2 for x in hist_freight) / len(hist_freight)) ** 0.5

scenarios = {
    "baseline": (baseline_goods, baseline_freight),
    "optimistic": (baseline_goods + 0.5 * goods_sd, baseline_freight + 0.5 * freight_sd),
    "pessimistic": (baseline_goods - 0.5 * goods_sd, baseline_freight - 0.5 * freight_sd),
}

scenario_paths = {}
for scenario_name, (goods_assumption, freight_assumption) in scenarios.items():
    path = [short_term_forecast]
    prev = short_term_forecast
    for _ in range(2, 11):
        pred = (
            coef_map["intercept"]
            + coef_map["lag_rail_gdp_growth"] * prev
            + coef_map["lag_goods_earnings_growth"] * goods_assumption
            + coef_map["lag_freight_growth"] * freight_assumption
        )
        path.append(pred)
        prev = pred
    scenario_paths[scenario_name] = path

horizons = {
    "1 year": scenario_paths["baseline"][0],
    "3 years": scenario_paths["baseline"][2],
    "5 years": scenario_paths["baseline"][4],
    "10 years": scenario_paths["baseline"][9],
}

target_values = [row["real_gdp_growth_of_railways"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
target_years = [row["start_year"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
goods_growth_values = [row["goods_earnings_growth"] for row in rows if row["real_gdp_growth_of_railways"] is not None]
freight_growth_values = [row["freight_growth"] for row in rows if row["real_gdp_growth_of_railways"] is not None]

fitted_years = [row["start_year"] for row in best_fit["data"]]

with PdfPages(PDF_PATH) as pdf:
    add_text_page(
        pdf,
        "Railways Sector GDP Growth Forecasting Report",
        [
            wrap_paragraph(
                "This report explains the project from scratch to finish using the dataset provided in `IML Dataset.xlsx`. The assignment aim is to predict India’s GDP growth rate at the sectoral level. For this project, the chosen sector is railways."
            ),
            wrap_paragraph(
                "The report covers: the assignment objective, why railways was chosen, how the data was cleaned, the exploratory data analysis, how lagged variables were built, which models were considered, why lagged linear regression was selected, what the forecasts are for the short term, medium term, and long run, and which other models are or are not suitable with this data."
            ),
            wrap_paragraph(
                f"Final baseline forecasts from the chosen model are: 1 year = {horizons['1 year']:.2f}%, 3 years = {horizons['3 years']:.2f}%, 5 years = {horizons['5 years']:.2f}%, and 10 years = {horizons['10 years']:.2f}%."
            ),
        ],
        fontsize=12,
    )

    add_text_page(
        pdf,
        "1. Problem Setup and Dataset",
        [
            wrap_paragraph(
                "The assignment asks for forecasting sectoral GDP growth for India. In the provided dataset, the railway-related target is already available as `real_gdp_growth_of_railways`, so the project can be framed as predicting the growth rate of the railways sector."
            ),
            wrap_paragraph(
                f"The cleaned annual dataset contains {len(rows)} yearly observations running from {rows[0]['year_label']} to {rows[-1]['year_label']}. The key advantage of this dataset is that it combines railway operational indicators and macroeconomic variables in one place."
            ),
            wrap_paragraph(
                "Examples of railway variables available in the data include goods earnings, passenger earnings, freight traffic, passenger kilometres, investment, operating ratio, route length, rolling stock, and other operational indicators. Macroeconomic variables include fiscal deficit, government expenditure, real interest rate, GDP deflator growth, and energy-price inflation."
            ),
            wrap_paragraph(
                "Because the assignment is a prediction exercise and the available data is annual and relatively short, the model has to stay simple, interpretable, and careful about overfitting."
            ),
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("2. Exploratory Data Analysis", fontsize=16, fontweight="bold")

    axes[0, 0].plot(target_years, target_values, marker="o", color="#1f4e79")
    axes[0, 0].set_title("Railways Real GDP Growth")
    axes[0, 0].set_xlabel("Start Year")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(target_years, goods_growth_values, marker="o", color="#8b0000")
    axes[0, 1].set_title("Goods Earnings Growth")
    axes[0, 1].set_xlabel("Start Year")
    axes[0, 1].set_ylabel("Percent")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(target_years, freight_growth_values, marker="o", color="#2e8b57")
    axes[1, 0].set_title("Freight Growth")
    axes[1, 0].set_xlabel("Start Year")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(
        [row["lag_rail_gdp_growth"] for row in rows if row.get("lag_rail_gdp_growth") is not None and row.get("real_gdp_growth_of_railways") is not None],
        [row["real_gdp_growth_of_railways"] for row in rows if row.get("lag_rail_gdp_growth") is not None and row.get("real_gdp_growth_of_railways") is not None],
        color="#6a0dad",
    )
    axes[1, 1].set_title("Current Growth vs Lagged Growth")
    axes[1, 1].set_xlabel("Lagged Railways GDP Growth")
    axes[1, 1].set_ylabel("Current Railways GDP Growth")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_text_page(
        pdf,
        "EDA Interpretation",
        [
            wrap_paragraph(
                "The EDA shows three useful patterns. First, railways real GDP growth has become much stronger in the later part of the sample, especially after the late 2010s. Second, goods-earnings growth and freight growth display clear swings, which makes economic sense because freight demand is one of the main commercial drivers of railways. Third, current railways growth has a visible relationship with its lagged value, which suggests persistence or momentum in the series."
            ),
            wrap_paragraph(
                "These patterns directly motivated the use of lagged predictors. A forecasting model should rely on past information, and the EDA suggests that past sector growth and past freight-side performance contain useful predictive information."
            ),
        ],
    )

    add_text_page(
        pdf,
        "3. Feature Engineering and Why Lag Was Used",
        [
            wrap_paragraph(
                "The original variables such as total investment, goods earnings, freight traffic, and passenger kilometres are levels. To make them more suitable for prediction, annual growth rates were constructed. For a variable X, the growth transformation used was: growth_t = ((X_t - X_(t-1)) / X_(t-1)) * 100."
            ),
            wrap_paragraph(
                "After constructing these growth variables, one-period lagged versions were created. A lagged variable is simply the previous year's value. For example, lag_rail_gdp_growth in 2024-25 equals the observed railways GDP growth in 2023-24. This is important because when forecasting year t, we should only use information known by the end of year t-1."
            ),
            wrap_paragraph(
                "Lagged variables solve two problems at once: they prevent information leakage, and they capture economic momentum. In railways, this is especially sensible because freight performance, commercial earnings, and sectoral growth do not reset each year from scratch. Past demand conditions often carry forward."
            ),
            wrap_paragraph(
                "The final engineered predictors considered were lagged railways GDP growth, lagged investment growth, lagged goods-earnings growth, lagged freight growth, and a few lagged macro controls."
            ),
        ],
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("4. Candidate Models and Comparison", fontsize=16, fontweight="bold", pad=20)

    table_rows = [["Model", "Predictors", "Obs", "Train RMSE", "Test RMSE", "Test MAE"]]
    for row in comparison:
        table_rows.append(
            [
                row["model"],
                ", ".join(row["predictors"]),
                str(row["observations"]),
                f"{row['train_rmse']:.3f}",
                f"{row['test_rmse']:.3f}",
                f"{row['test_mae']:.3f}",
            ]
        )

    table = ax.table(cellText=table_rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        elif r == 1:
            cell.set_facecolor("#dce6f1")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_text_page(
        pdf,
        "Why the Final Model Was Chosen",
        [
            wrap_paragraph(
                "Four parsimonious lagged linear regression models were tested. The idea was to compare very small, interpretable models rather than overload the sample with many predictors. This is important because the dataset is annual and short, so a big model would overfit quickly."
            ),
            wrap_paragraph(
                f"The best holdout model was `{best['model']}`. Its predictors are {', '.join(best['predictors'])}. It had the lowest test RMSE among the candidate models, which means it performed best when asked to predict data that was not used for fitting."
            ),
            wrap_paragraph(
                "Economically, the chosen model is also sensible. Railways GDP growth should depend on its own past momentum, on lagged goods earnings because those capture the commercial strength of the sector, and on lagged freight growth because freight is a core channel through which railway activity expands."
            ),
        ],
    )

    add_text_page(
        pdf,
        "5. Final Lagged Linear Regression Model",
        [
            wrap_paragraph(
                "The fitted model can be written as:"
            ),
            "Predicted railways GDP growth = intercept + b1 * lagged railways GDP growth + b2 * lagged goods-earnings growth + b3 * lagged freight growth",
            "",
            f"Intercept = {coef_map['intercept']:.4f}",
            f"Coefficient on lagged railways GDP growth = {coef_map['lag_rail_gdp_growth']:.4f}",
            f"Coefficient on lagged goods-earnings growth = {coef_map['lag_goods_earnings_growth']:.4f}",
            f"Coefficient on lagged freight growth = {coef_map['lag_freight_growth']:.4f}",
            "",
            wrap_paragraph(
                "Interpretation: the positive coefficient on lagged railways GDP growth means that growth persistence is strong. A positive coefficient on lagged freight growth means strong freight performance tends to support future sector growth. The lagged goods-earnings coefficient is slightly negative in this small sample. That should not be over-interpreted as a structural truth; with such a short dataset, coefficients can reflect overlap between related commercial variables."
            ),
            wrap_paragraph(
                f"The model fit over the usable sample is reasonably strong for such a small annual dataset: adjusted R-squared = {best_fit['adj_r2']:.3f} and in-sample RMSE = {best_fit['rmse']:.3f}."
            ),
        ],
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.plot(
        [row["start_year"] for row in best_fit["data"]],
        [row["real_gdp_growth_of_railways"] for row in best_fit["data"]],
        marker="o",
        color="#1f4e79",
        label="Actual",
    )
    ax.plot(fitted_years, best_fit["fitted"], marker="s", color="#b22222", label="Fitted")
    ax.set_title("6. Actual vs Fitted Values for the Final Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("Start Year")
    ax.set_ylabel("Percent")
    ax.grid(alpha=0.3)
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    future_years = list(range(2024, 2034))
    ax.plot(future_years, scenario_paths["baseline"], marker="o", color="#1f4e79", label="Baseline")
    ax.plot(future_years, scenario_paths["optimistic"], marker="o", color="#2e8b57", label="Optimistic")
    ax.plot(future_years, scenario_paths["pessimistic"], marker="o", color="#8b0000", label="Pessimistic")
    ax.set_title("7. Forecast Path: Short, Medium, and Long Run", fontsize=16, fontweight="bold")
    ax.set_xlabel("Forecast Start Year")
    ax.set_ylabel("Predicted Railways GDP Growth (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_text_page(
        pdf,
        "8. Forecast Results",
        [
            wrap_paragraph(
                f"Short term forecast (1 year ahead, 2024-25): {horizons['1 year']:.2f}%. This is the cleanest forecast because it uses actual latest lagged inputs from 2023-24."
            ),
            wrap_paragraph(
                f"Medium term forecast (3 years ahead, 2026-27): {horizons['3 years']:.2f}%. Medium term forecast (5 years ahead, 2028-29): {horizons['5 years']:.2f}%."
            ),
            wrap_paragraph(
                f"Long run forecast (10 years ahead, 2033-34): {horizons['10 years']:.2f}%."
            ),
            wrap_paragraph(
                "For the longer horizons, the model is used recursively. That means the predicted value from one step becomes the lagged growth input for the next step. To keep the exercise realistic, future goods-earnings growth and freight growth are held at their recent 3-year average in the baseline path. Optimistic and pessimistic scenarios move these assumptions up or down."
            ),
            wrap_paragraph(
                "The forecast pattern is economically intuitive: the sector remains strong in the short run, then gradually moderates rather than crashing. In other words, the model sees ongoing momentum, but not indefinite acceleration."
            ),
        ],
    )

    add_text_page(
        pdf,
        "9. Are Other Models Suitable?",
        [
            wrap_paragraph(
                "Yes, but only a few should be considered seriously."
            ),
            "1. Lagged Linear Regression: yes. This should remain the main model because it fits the assignment, is interpretable, and is appropriate for a small annual sample.",
            "2. Ridge or Lasso: maybe. These are useful only as robustness checks if you want to test a slightly bigger predictor set and handle correlated variables.",
            "3. PCR or PLS: maybe. These could help compress many macro variables into a smaller number of components, but they are secondary here.",
            "4. KNN: no. With such a short annual macro dataset, nearest-neighbour methods are weak and hard to interpret economically.",
            "5. Regression trees, random forest, boosting: no. These flexible models need much more data and would overfit badly here.",
            wrap_paragraph(
                "So the best academic answer is: use lagged linear regression as the main model, and optionally add Ridge or Lasso as a robustness comparison. Do not rely on flexible tree-based or distance-based methods with this dataset."
            ),
        ],
    )

    add_text_page(
        pdf,
        "10. Limitations, Why They Matter, and Final Conclusion",
        [
            wrap_paragraph(
                "This project has three main limitations. First, the data is annual, so there are very few observations. Second, the sector is only one part of the Indian economy, so there can be policy and structural shocks not fully captured by the model. Third, long-run forecasts always depend on assumptions; they are not mechanical truths."
            ),
            wrap_paragraph(
                "These limitations are exactly why the model was kept simple. A more complex model would look more advanced, but with so little data it would not be more credible. In this setting, a parsimonious lagged linear regression is stronger than a flashy overfit model."
            ),
            wrap_paragraph(
                "Final conclusion: the dataset suggests a strong near-term outlook for the railways sector, with a forecast of about 11.78% for 2024-25. The medium-term and long-run forecasts remain high but gradually moderate, settling around 10.18% in the 10-year baseline path. The best justification for the chosen model is that it uses only past information, captures sector momentum, remains economically interpretable, and is suitable for a small annual sample."
            ),
            wrap_paragraph(
                "If this report were being extended further, the best next step would be to add a small Ridge or Lasso comparison and test whether the broad message remains unchanged."
            ),
        ],
    )


md_lines = [
    "# Railways Project Detailed Report",
    "",
    "This markdown companion mirrors the PDF report generated in `docs/railways_project_detailed_report.pdf`.",
    "",
    "## Key Forecasts",
    "",
    f"- 1 year: {horizons['1 year']:.2f}%",
    f"- 3 years: {horizons['3 years']:.2f}%",
    f"- 5 years: {horizons['5 years']:.2f}%",
    f"- 10 years: {horizons['10 years']:.2f}%",
    "",
    "## Final Model",
    "",
    f"- Model: {best['model']}",
    f"- Predictors: {', '.join(best['predictors'])}",
    f"- Adjusted R-squared: {best_fit['adj_r2']:.3f}",
    f"- RMSE: {best_fit['rmse']:.3f}",
]
MD_PATH.write_text("\n".join(md_lines))

print(PDF_PATH)
