from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"
PDF_PATH = DOCS_DIR / "allvars_workflow_detailed_report.pdf"
MD_PATH = DOCS_DIR / "allvars_workflow_detailed_report.md"

DOCS_DIR.mkdir(parents=True, exist_ok=True)


def wrap(text: str, width: int = 95) -> str:
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


def add_table_page(pdf: PdfPages, title: str, dataframe: pd.DataFrame, max_rows: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    df = dataframe.copy().fillna("")
    if max_rows is not None:
        df = df.head(max_rows)
    table_rows = [list(df.columns)] + df.astype(str).values.tolist()
    table = ax.table(cellText=table_rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


current_selection = pd.read_csv(OUTPUT_DIR / "allvars_current_lasso_selection.csv")
lagged_selection = pd.read_csv(OUTPUT_DIR / "allvars_lagged_lasso_selection.csv")
current_summary = pd.read_csv(OUTPUT_DIR / "allvars_current_model_summary.csv")
lagged_summary = pd.read_csv(OUTPUT_DIR / "allvars_lagged_model_summary.csv")
current_prediction = pd.read_csv(OUTPUT_DIR / "allvars_current_2024_25_prediction.csv")
lagged_prediction = pd.read_csv(OUTPUT_DIR / "allvars_lagged_2024_25_prediction.csv")
lagged_horizons = pd.read_csv(OUTPUT_DIR / "allvars_lagged_horizon_forecasts.csv")
current_pca = pd.read_csv(OUTPUT_DIR / "allvars_current_pca_summary.csv")
lagged_pca = pd.read_csv(OUTPUT_DIR / "allvars_lagged_pca_summary.csv")
lagged_scenarios = pd.read_csv(OUTPUT_DIR / "allvars_lagged_scenario_forecast_path.csv")

selected_current = current_selection.loc[current_selection["selected"] == "yes", "variable"].tolist()
selected_lagged = lagged_selection.loc[lagged_selection["selected"] == "yes", "variable"].tolist()


with PdfPages(PDF_PATH) as pdf:
    add_text_page(
        pdf,
        "Detailed Report: All-Variable Lasso, PCA, and Forecasting Workflow",
        [
            wrap(
                "This report explains the all-variable workflow in detail. Unlike the earlier screened-candidate approach, this version lets Lasso see the entire cleaned numeric dataset rather than a manually narrowed pool. The purpose is to answer the methodological question: what happens if formal shrinkage and selection are allowed to operate on almost the whole dataset?"
            ),
            wrap(
                "The workflow still follows the professor’s requested structure: first use Lasso, then use PCA, then estimate one current-variable model for 2024-25 and one lagged-variable model for forecasting. The difference is that here Lasso gets the full cleaned design matrix rather than a hand-filtered subset of railway and macro variables."
            ),
            wrap(
                f"The final headline outputs are: current-year nowcast for 2024-25 = {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%; lagged forecast for 2024-25 = {lagged_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%; lagged 5-year forecast = {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%; lagged 10-year forecast = {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%."
            ),
        ],
        fontsize=12,
    )

    add_text_page(
        pdf,
        "1. What 'All Variables' Means in This Workflow",
        [
            wrap(
                "All variables does not mean every raw cell from the spreadsheet with no processing. It means all cleaned numeric columns that can reasonably enter a model after basic preparation. The cleaned dataset contains numeric railway operating variables, financial variables, freight and passenger variables, infrastructure variables, and macroeconomic variables."
            ),
            wrap(
                "For the current-year branch, every numeric predictor except the target itself was made available to Lasso. That produced a pool of 60 current-year numeric candidate predictors. For the lagged branch, lagged versions of all cleaned numeric variables were created, including a lagged version of the target, and Lasso then saw 61 lagged candidate predictors."
            ),
            wrap(
                "This approach is much more automatic than screened-candidate modeling. It minimizes human pre-selection, but it also creates risks: many variables are highly correlated, some are noisy, some are slow-moving stock variables, and some may be only indirectly related to the forecasting target."
            ),
        ],
    )

    add_text_page(
        pdf,
        "2. Pre-Lasso Preparation Steps",
        [
            wrap(
                "Before Lasso can be used on a full dataset, a few non-negotiable preprocessing steps are required. First, non-numeric columns such as the year label must be excluded. Second, the target variable must not be included as a current-year predictor in the current branch. Third, because the dataset contains missing values, imputation must be added so that Lasso can still process the full matrix."
            ),
            wrap(
                "Median imputation was used before scaling. This is important because Lasso cannot be estimated with missing values left in place. Standardization was also applied because Lasso penalties are sensitive to scale; variables measured in large units would otherwise dominate variables measured in small units."
            ),
            wrap(
                "For the lagged branch, the same preparation logic applied after constructing one-period lags. In that branch, the lagged target variable was allowed to enter because past railways GDP growth is a valid forecasting predictor."
            ),
        ],
    )

    add_table_page(pdf, "3. Current-Variable Lasso Selection from the Full Dataset", current_selection, max_rows=25)
    add_text_page(
        pdf,
        "How Lasso Behaved in the Current-Variable Branch",
        [
            wrap(
                "In the current-year branch, Lasso started from 60 current-year numeric candidate variables. After shrinkage and cross-validation, it kept 14 variables and shrank the rest to zero."
            ),
            wrap(
                f"The selected variables were: {', '.join(selected_current)}."
            ),
            wrap(
                "This is a very different result from the screened-candidate workflow. Because Lasso was free to scan the full numeric matrix, it selected a mix of infrastructure, traffic, labour, freight, macro, and contribution variables. This is exactly the kind of output that can happen when a wide dataset is given to an automated selector with a short sample."
            ),
            wrap(
                "The advantage is methodological purity: the modeler cannot be accused of hand-picking a small preferred set. The disadvantage is that the resulting variable set is harder to interpret economically, because it combines many types of signals and may reflect small-sample quirks."
            ),
        ],
    )

    add_table_page(pdf, "4. PCA Summary for the Current-Variable Branch", current_pca)
    add_text_page(
        pdf,
        "Why PCA Was Useful in the Current Branch",
        [
            wrap(
                "Once Lasso kept 14 current-year variables, PCA became meaningful. At that point, there really was a block of selected variables with overlapping information. PCA reduced that selected set into 6 principal components."
            ),
            wrap(
                "This is a good example of when PCA becomes useful. Unlike the earlier professor workflow on screened variables, where Lasso left too little macro structure for PCA to act on, the all-variable workflow gave PCA a real compression task."
            ),
            wrap(
                "The principal components are not individually as interpretable as the original variables, but they summarize the common structure in the selected set. So the current final regression became a regression on 6 PCs rather than on 14 original variables."
            ),
        ],
    )

    add_table_page(pdf, "5. Current-Variable Model Summary", current_summary)
    add_text_page(
        pdf,
        "How the Current-Year Model Was Used",
        [
            wrap(
                "After Lasso and PCA, a linear regression was fitted using the principal components from the selected current-year variables. The purpose of this model is not strict forecasting but current-year prediction. It uses same-year explanatory variables to estimate same-year 2024-25 railways GDP growth."
            ),
            wrap(
                f"The current-year nowcast produced by this all-variable workflow was {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%. The in-sample R-squared is high at {current_summary.loc[0, 'r_squared']:.3f}, and the RMSE is {current_summary.loc[0, 'rmse']:.3f}."
            ),
            wrap(
                "The relatively strong in-sample fit is not surprising because the model is using many current-year signals condensed through PCA. But this does not automatically make it a better economic model. High in-sample fit in a small sample can be partly mechanical."
            ),
        ],
    )

    add_table_page(pdf, "6. Lagged-Variable Lasso Selection from the Full Dataset", lagged_selection, max_rows=25)
    add_text_page(
        pdf,
        "How Lasso Behaved in the Lagged Branch",
        [
            wrap(
                "In the lagged branch, Lasso started from 61 lagged candidate variables. After automatic shrinkage, it kept 7 lagged predictors."
            ),
            wrap(
                f"The selected lagged variables were: {', '.join(selected_lagged)}."
            ),
            wrap(
                "Again, the selected set is more eclectic than in the screened-candidate workflow. It includes lagged infrastructure, lagged capital, lagged operating ratio, lagged iron-ore traffic, lagged passenger kilometres, lagged employment, and lagged GDP-deflator growth."
            ),
            wrap(
                "This outcome is very instructive. When Lasso sees the full lagged matrix, it does not necessarily pick the variables that are easiest to justify economically. It picks variables that help prediction under the penalty, and in a short annual sample that can lead to unusual combinations."
            ),
        ],
    )

    add_table_page(pdf, "7. PCA Summary for the Lagged Branch", lagged_pca)
    add_text_page(
        pdf,
        "Why PCA Was Also Useful in the Lagged Branch",
        [
            wrap(
                "Because Lasso left 7 lagged variables, PCA again had a real role to play. It reduced the selected lagged variables into 4 principal components."
            ),
            wrap(
                "This means the final lagged forecasting regression was not run on the raw selected lagged variables directly. Instead, it was run on the 4 PCs that summarized them. That helps reduce collinearity and dimensionality, but it also makes the final model more abstract and less directly interpretable."
            ),
            wrap(
                "This is the natural tradeoff in the all-variable workflow: less human screening, more automatic compression, and lower interpretability of the end model."
            ),
        ],
    )

    add_table_page(pdf, "8. Lagged-Variable Model Summary", lagged_summary)
    add_text_page(
        pdf,
        "How the Lagged Model Was Used",
        [
            wrap(
                "The lagged branch is the true forecasting branch. Every predictor is lagged, meaning the model uses past information only. That makes it the appropriate branch for next-year, 5-year, and 10-year prediction."
            ),
            wrap(
                f"The one-step-ahead forecast for 2024-25 was {lagged_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%. The 5-year baseline forecast was {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%, and the 10-year baseline forecast was {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%."
            ),
            wrap(
                f"However, the holdout performance is noticeably weaker than in the smaller screened-candidate lagged model. The test RMSE here is {lagged_summary.loc[0, 'test_rmse']:.3f}, which is substantially higher than the earlier more curated lagged model. This is a very important finding."
            ),
        ],
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    for scenario, color in [("baseline", "#1f4e79"), ("optimistic", "#2e8b57"), ("pessimistic", "#8b0000")]:
        subset = lagged_scenarios[lagged_scenarios["scenario"] == scenario]
        ax.plot(subset["forecast_year"], subset["predicted_real_gdp_growth_of_railways"], marker="o", label=scenario.title(), color=color)
    ax.set_title("9. All-Variable Lagged Forecast Paths", fontsize=16, fontweight="bold")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Predicted Railways GDP Growth (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_table_page(pdf, "10. Key Forecast Horizons from the All-Variable Lagged Model", lagged_horizons)
    add_text_page(
        pdf,
        "How to Read These Forecasts",
        [
            wrap(
                "The all-variable lagged forecast path rises quickly and then flattens. That is why the 5-year and 10-year forecasts are both around 16.10%. This happens because recursive forecasting with principal components can settle into a stable point depending on the structure of the fitted model."
            ),
            wrap(
                "This is mathematically possible, but economically it should be treated with caution. A very smooth convergence to a high value does not necessarily mean that the railways sector will truly grow at that exact long-run rate. It means that given the selected lagged principal-component structure, the recursive system converges in that direction."
            ),
            wrap(
                "That is another reason why the all-variable workflow should be interpreted carefully. It is rigorous in one sense, because it lets formal selection and compression drive the process. But it is also more fragile, because the selected variables and components may reflect small-sample patterns that are not strong economic laws."
            ),
        ],
    )

    add_text_page(
        pdf,
        "11. Strengths of the All-Variable Workflow",
        [
            wrap(
                "The biggest strength is transparency of selection. The modeler is not pre-deciding a small favourite set of variables before Lasso. Instead, Lasso sees the full cleaned numeric dataset, so the selected set is more purely data-driven."
            ),
            wrap(
                "A second strength is that PCA genuinely becomes active in this workflow. Because Lasso keeps larger variable sets, PCA has enough structure to compress. This creates a workflow that follows the professor’s requested order in a very literal way: Lasso first, then PCA, then final prediction."
            ),
            wrap(
                "A third strength is methodological completeness. This workflow is useful as a robustness exercise or methodological appendix because it shows what the model would do under a very broad and automatic selection setup."
            ),
        ],
    )

    add_text_page(
        pdf,
        "12. Weaknesses and Risks of the All-Variable Workflow",
        [
            wrap(
                "The main weakness is interpretability. When Lasso is given the whole numeric dataset, it may select variables that are statistically useful in-sample but difficult to justify economically. That is exactly what happened here. The selected variables are broader and stranger than in the screened-candidate workflow."
            ),
            wrap(
                "A second weakness is instability in a short annual sample. Because the dataset is not large, different automatic-selection paths can produce very different chosen variables. The more variables one gives to the selector, the larger the risk that it will pick up noise or collinearity artifacts."
            ),
            wrap(
                "A third weakness is predictive reliability in the lagged branch. The all-variable lagged model has worse holdout error than the earlier smaller lagged model. That suggests that complete automation is not automatically better forecasting."
            ),
            wrap(
                "So although this workflow is methodologically appealing in one sense, it should not automatically be treated as the best final forecasting model."
            ),
        ],
    )

    add_text_page(
        pdf,
        "13. How This Differs from the Earlier Screened Workflow",
        [
            wrap(
                "In the earlier screened workflow, the candidate set was intentionally constrained to economically meaningful railway and macro variables before Lasso was run. That meant the automatic selector worked inside a curated economic frame."
            ),
            wrap(
                "In the all-variable workflow, that screening step was mostly removed. Lasso saw almost the whole cleaned numeric matrix. As a result, PCA became more active, but the selected variables became less interpretable and the lagged holdout performance deteriorated."
            ),
            wrap(
                "This comparison is extremely useful for your final write-up. It shows that the modeling choice is not simply about what is most automatic. It is about balancing formal selection, economic meaning, and predictive stability."
            ),
        ],
    )

    add_text_page(
        pdf,
        "14. Final Conclusion",
        [
            wrap(
                "The all-variable workflow was successfully implemented exactly as requested: all cleaned numeric variables were given to Lasso, PCA was then applied to the selected sets, and separate current-year and lagged forecasting models were estimated."
            ),
            wrap(
                f"The current-year nowcast from this workflow is {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%. The lagged forecasts are {lagged_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}% for 2024-25, {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}% for the 5-year horizon, and {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}% for the 10-year horizon."
            ),
            wrap(
                "But the most important result is not just the forecast numbers. It is the methodological lesson: letting Lasso see the whole dataset makes the process more automatic, but it also makes the resulting model less interpretable and, in the lagged branch, less reliable out of sample."
            ),
            wrap(
                "That means this all-variable workflow is valuable as a robustness exercise and as a direct answer to the professor’s methodological question. For a final defensible submission, it should probably be presented alongside the screened-candidate workflow rather than replacing it entirely."
            ),
        ],
    )


md_lines = [
    "# All-Variable Workflow Detailed Report",
    "",
    f"- Current-year nowcast: {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- Lagged 2024-25 forecast: {lagged_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- Lagged 5-year forecast: {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%",
    f"- Lagged 10-year forecast: {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%",
    "",
    f"- Current selected count: {len(selected_current)}",
    f"- Lagged selected count: {len(selected_lagged)}",
]
MD_PATH.write_text("\n".join(md_lines))

print(PDF_PATH)
