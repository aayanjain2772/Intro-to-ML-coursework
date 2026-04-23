from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
DOCS_DIR = REPO_ROOT / "docs"
PDF_PATH = DOCS_DIR / "professor_workflow_detailed_report.pdf"
MD_PATH = DOCS_DIR / "professor_workflow_detailed_report.md"

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


def add_table_page(pdf: PdfPages, title: str, dataframe: pd.DataFrame, highlight_row: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    df = dataframe.copy()
    df = df.fillna("")
    table_rows = [list(df.columns)] + df.astype(str).values.tolist()
    table = ax.table(cellText=table_rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        elif highlight_row is not None and r == highlight_row + 1:
            cell.set_facecolor("#dce6f1")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


df = pd.read_csv(REPO_ROOT / "data" / "processed" / "railways_master.csv")
current_selection = pd.read_csv(OUTPUT_DIR / "prof_current_lasso_selection.csv")
lagged_selection = pd.read_csv(OUTPUT_DIR / "prof_lagged_lasso_selection.csv")
current_summary = pd.read_csv(OUTPUT_DIR / "prof_current_model_summary.csv")
lagged_summary = pd.read_csv(OUTPUT_DIR / "prof_lagged_model_summary.csv")
current_prediction = pd.read_csv(OUTPUT_DIR / "prof_current_2024_25_prediction.csv")
lagged_horizons = pd.read_csv(OUTPUT_DIR / "prof_lagged_horizon_forecasts.csv")
lagged_scenarios = pd.read_csv(OUTPUT_DIR / "prof_lagged_scenario_forecast_path.csv")
current_coefficients = pd.read_csv(OUTPUT_DIR / "prof_current_model_coefficients.csv")
lagged_coefficients = pd.read_csv(OUTPUT_DIR / "prof_lagged_model_coefficients.csv")

selected_current = current_selection.loc[current_selection["selected"] == "yes", "variable"].tolist()
selected_lagged = lagged_selection.loc[lagged_selection["selected"] == "yes", "variable"].tolist()

with PdfPages(PDF_PATH) as pdf:
    add_text_page(
        pdf,
        "Detailed Report: Lasso, PCA, Current Model, and Lagged Model",
        [
            wrap(
                "This report explains the full workflow requested by the professor. The process had four steps: first use Lasso for variable selection, second apply PCA where appropriate, third build a current-variable model for predicting the 2024-25 railways GDP growth rate, and fourth build a lagged-variable model for medium-run and long-run forecasting."
            ),
            wrap(
                "The chosen sector is railways, and the target variable in the dataset is `real_gdp_growth_of_railways`. The report explains not only what was done, but why each step was done, what the technical logic was, what the result of each step was, and how the two final models differ conceptually."
            ),
            wrap(
                f"The final headline results are: current-year nowcast for 2024-25 = {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%; lagged-model forecast for 2024-25 = {lagged_horizons.loc[lagged_horizons['horizon']=='short_term_1_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%; 5-year lagged forecast = {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%; 10-year lagged forecast = {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%."
            ),
        ],
        fontsize=12,
    )

    add_text_page(
        pdf,
        "1. Why the Workflow Was Structured This Way",
        [
            wrap(
                "The dataset is wide, meaning it contains many possible railway and macroeconomic variables, but it is not deep in time because the sample is annual. That creates a standard modeling problem: there are too many candidate predictors relative to the number of observations. If all variables were thrown into one regression directly, the model would be unstable and likely overfit."
            ),
            wrap(
                "Because of that, the workflow was designed in a disciplined sequence. Lasso came first because it is a formal course-covered way to reduce the predictor set. PCA came next because if several macro variables survived Lasso, they could still be correlated with each other, and PCA would compress that macro block into a smaller number of orthogonal components. Only after those dimensionality-reduction steps should the final predictive regressions be estimated."
            ),
            wrap(
                "The final modeling structure deliberately separates a current-year model from a lagged forecasting model. This distinction matters. A model using current-year explanatory variables is not a pure forecast; it is better described as a contemporaneous prediction model or nowcast. A model using lagged values is the genuine forecasting model because it relies only on past information."
            ),
        ],
    )

    add_text_page(
        pdf,
        "2. Data Preparation and Candidate Variables",
        [
            wrap(
                "The raw railway dataset was first cleaned into a processed annual CSV. From that file, the target variable remained `real_gdp_growth_of_railways`. Several level variables such as total investment, goods earnings, freight traffic, passenger kilometres, and working expenditure were converted into annual percentage growth rates. This was done because the target is also a growth rate, so using growth-based predictors improves comparability and avoids simply regressing one trending level on another."
            ),
            wrap(
                "For the current-variable model, the candidate set was intentionally restricted to variables that are both economically meaningful and available for 2024-25 in the dataset. Those candidates were: investment growth, goods earnings growth, freight growth, operating ratio, fiscal deficit, government expenditure, and GDP deflator growth."
            ),
            wrap(
                "For the lagged model, the candidate set was formed by taking lagged versions of meaningful railway and macro variables. Those included lagged railways GDP growth, lagged investment growth, lagged goods-earnings growth, lagged freight growth, lagged operating ratio, lagged fiscal deficit, lagged government expenditure, and lagged GDP deflator growth."
            ),
            wrap(
                "This candidate-pool step is important. Lasso is a formal selector, but it should not be fed every raw column blindly. The modeler still has to define a sensible economic pool before automated selection begins."
            ),
        ],
    )

    add_table_page(pdf, "3. Lasso Selection for the Current-Variable Model", current_selection)
    add_text_page(
        pdf,
        "What Was Done in the Current-Variable Lasso Step",
        [
            wrap(
                "Lasso was used here because the model still had several current-year railway and macro candidates. Lasso works by estimating a regression with a penalty on coefficient size. Weak or redundant variables get pushed toward zero. When the penalty is strong enough, some coefficients become exactly zero. This creates a formal variable-selection rule."
            ),
            wrap(
                "Cross-validated Lasso was used, which means the penalty parameter was chosen by testing different penalty strengths and selecting the one that balances fit and parsimony. This is better than choosing the penalty manually."
            ),
            wrap(
                f"In the current-variable branch, Lasso selected two variables: {', '.join(selected_current)}. Everything else was shrunk to zero. This means that among the available same-year candidates, these two variables contained the strongest predictive signal for the target."
            ),
            wrap(
                "Economically, that is plausible. Operating ratio measures railway operating efficiency and cost pressure, while fiscal deficit captures the wider fiscal environment. Both can be relevant in explaining same-year sector performance."
            ),
        ],
    )

    add_text_page(
        pdf,
        "4. PCA in the Current-Variable Branch",
        [
            wrap(
                "PCA was placed after Lasso because the goal was not to compress everything, but specifically to compress any remaining macro block if multiple macro variables survived selection. PCA reduces dimensionality by replacing correlated variables with principal components that summarize most of the variation."
            ),
            wrap(
                "In this current-variable branch, only one macro variable survived Lasso: `fiscal_deficit_gdp`. Since PCA requires at least a small block of variables to be meaningful, there was no multi-variable macro block left to compress. Therefore, PCA was checked, but it did not create any component in the final current model."
            ),
            wrap(
                "This is not a failure of PCA. It is a result of the sequencing. Lasso had already simplified the data enough that PCA no longer had a real job to do in this branch. The correct thing statistically is to keep the surviving macro variable directly rather than pretend a principal component exists when it does not."
            ),
        ],
    )

    add_table_page(pdf, "5. Current-Variable Model Summary", current_summary)
    add_table_page(pdf, "6. Current-Variable Model Coefficients", current_coefficients)
    add_text_page(
        pdf,
        "How the Current-Variable Model Was Estimated",
        [
            wrap(
                "After Lasso and the PCA check, the final current-variable model was an ordinary linear regression using the variables that survived the pipeline. In this case, the final predictors were `operating_ratio` and `fiscal_deficit_gdp`."
            ),
            wrap(
                "This model uses current-year values to explain current-year railways GDP growth. That means the 2024-25 prediction generated by this model is a nowcast or contemporaneous estimate, not a strict out-of-sample forecast. The reason for doing this model anyway is that your professor explicitly asked for a same-year explanatory model, and such a model can be useful for estimating the likely sector outcome once current-year covariates are observed."
            ),
            wrap(
                f"The predicted 2024-25 railways GDP growth from this branch was {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%. The model fit over the usable historical sample had R-squared = {current_summary.loc[0, 'r_squared']:.3f} and RMSE = {current_summary.loc[0, 'rmse']:.3f}."
            ),
            wrap(
                "The interpretation is straightforward: the current-year model says that once you know the same-year operating ratio and fiscal deficit conditions, the implied 2024-25 railways growth rate is about 8.04 percent."
            ),
        ],
    )

    add_table_page(pdf, "7. Lasso Selection for the Lagged Forecasting Model", lagged_selection)
    add_text_page(
        pdf,
        "What Was Done in the Lagged Lasso Step",
        [
            wrap(
                "The lagged branch is the true forecasting branch. Every predictor was lagged so that the model uses past information only. This is exactly what should happen in a genuine forecast. For example, to predict 2024-25 railways GDP growth, the model uses 2023-24 lagged values rather than 2024-25 same-year values."
            ),
            wrap(
                "The same Lasso logic was applied here. A lagged candidate pool was formed, then cross-validated Lasso was used to shrink weak lagged variables to zero."
            ),
            wrap(
                f"In the lagged branch, Lasso selected two variables: {', '.join(selected_lagged)}. That means the strongest lagged predictors were past railways GDP growth itself and lagged operating ratio."
            ),
            wrap(
                "Economically, this is sensible. Lagged railways GDP growth captures persistence or momentum in the sector. Lagged operating ratio captures operational efficiency and cost discipline from the prior year, which can carry into future performance."
            ),
        ],
    )

    add_text_page(
        pdf,
        "8. PCA in the Lagged Branch",
        [
            wrap(
                "PCA was again evaluated after Lasso. However, in the lagged branch, no multi-variable lagged macro block survived Lasso. Because of that, PCA did not add any principal components in the final lagged forecasting model."
            ),
            wrap(
                "This is an important detail to explain in the report and viva. The workflow still included PCA conceptually, but PCA only becomes active when there is a sufficiently large correlated block to compress. Here, Lasso had already simplified the structure so much that PCA had no remaining correlated macro block to summarize."
            ),
            wrap(
                "So the honest explanation is: PCA was part of the planned method, it was checked, but after Lasso there were not enough selected macro variables for PCA to meaningfully operate in the final lagged model."
            ),
        ],
    )

    add_table_page(pdf, "9. Lagged Forecasting Model Summary", lagged_summary)
    add_table_page(pdf, "10. Lagged Forecasting Model Coefficients", lagged_coefficients)
    add_text_page(
        pdf,
        "How the Lagged Forecasting Model Was Estimated",
        [
            wrap(
                "Once Lasso had selected the final lagged predictors and PCA had been checked, a lagged linear regression was estimated on the selected variables. The final predictors were `lag_rail_gdp_growth` and `lag_operating_ratio`."
            ),
            wrap(
                f"This model performed quite well on the holdout period for such a small annual sample. The training RMSE was {lagged_summary.loc[0, 'train_rmse']:.3f}, and the test RMSE was {lagged_summary.loc[0, 'test_rmse']:.3f}. That lower test error is one reason this branch is more suitable for actual forecasting than the current-year branch."
            ),
            wrap(
                "The 2024-25 one-step-ahead forecast from the lagged model was produced using actual lagged 2023-24 values. For the longer horizons, the model was used recursively. That means the predicted value from one step becomes the lagged GDP-growth input for the next step, while the other selected lagged exogenous variable follows scenario assumptions."
            ),
        ],
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    for scenario, color in [("baseline", "#1f4e79"), ("optimistic", "#2e8b57"), ("pessimistic", "#8b0000")]:
        subset = lagged_scenarios[lagged_scenarios["scenario"] == scenario]
        ax.plot(subset["forecast_year"], subset["predicted_real_gdp_growth_of_railways"], marker="o", label=scenario.title(), color=color)
    ax.set_title("11. Lagged Model Forecast Paths", fontsize=16, fontweight="bold")
    ax.set_xlabel("Forecast Year")
    ax.set_ylabel("Predicted Railways GDP Growth (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_table_page(pdf, "12. Key Lagged Forecast Horizons", lagged_horizons)
    add_text_page(
        pdf,
        "Forecast Interpretation",
        [
            wrap(
                f"The lagged model forecast for 2024-25 is {lagged_horizons.loc[lagged_horizons['horizon']=='short_term_1_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%. The 5-year baseline forecast is {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%, and the 10-year baseline forecast is {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%."
            ),
            wrap(
                "The shape of the forecast path is relatively stable because the selected lagged variables imply persistence in railways growth and a continuing role for operating performance. Since there are few predictors, the longer-horizon path is smooth rather than volatile."
            ),
            wrap(
                "This is one major difference from the current-year model. The current-year model gives one contemporaneous estimate for 2024-25, while the lagged model generates a path through time and is therefore the appropriate tool for medium-run and long-run forecasting."
            ),
        ],
    )

    add_text_page(
        pdf,
        "13. Exactly How the Two Models Differ",
        [
            "Current-variable model:",
            wrap(
                "Uses same-year predictors to estimate same-year 2024-25 railways GDP growth. It is a nowcasting or contemporaneous explanatory model. It answers: given current-year conditions, what does the sector growth rate look like for 2024-25?"
            ),
            "",
            "Lagged-variable model:",
            wrap(
                "Uses previous-year predictors to forecast future railways GDP growth. It is the genuine forecasting model. It answers: using only past information, what growth should we expect next year, in five years, and in ten years?"
            ),
            "",
            "Why this difference matters:",
            wrap(
                "The current model is useful when current-year explanatory variables are already observed. The lagged model is useful when forecasting future values without access to same-year explanatory data."
            ),
            "",
            "Why the outputs differ:",
            wrap(
                "They differ because they are solving different statistical problems. The current model uses contemporaneous information and therefore estimates a same-year value. The lagged model uses only lagged values and therefore captures momentum and delayed operational effects instead."
            ),
        ],
    )

    add_text_page(
        pdf,
        "14. What Was Missing or Important to Add in the Process",
        [
            wrap(
                "Several process details are easy to miss but matter a lot. First, the current-year branch had to be restricted to variables that were actually available for 2024-25. Some seemingly useful variables, such as passenger kilometres or real interest rate, had missing values in the 2024-25 row and therefore could not be used cleanly in the nowcast model."
            ),
            wrap(
                "Second, after Lasso selection, PCA must be interpreted honestly. It is not enough to say that PCA was used. One has to say whether PCA actually produced components in the final selected set. In this project, PCA was evaluated but did not produce additional components because the selected macro block became too small."
            ),
            wrap(
                "Third, after selection steps such as Lasso, the final linear regression was re-estimated on the selected predictors to keep the end model interpretable. That is important because Lasso is ideal for selection, but plain linear regression is easier to explain when writing down coefficients and interpreting economic effects."
            ),
            wrap(
                "Fourth, the longer-run forecasts were generated recursively. This should always be stated clearly because a 10-year forecast is not one direct shot from the data; it is the result of repeatedly feeding prior predicted values into future steps."
            ),
        ],
    )

    add_text_page(
        pdf,
        "15. Final Conclusion",
        [
            wrap(
                "The professor-requested workflow was completed exactly in spirit: Lasso was used first for formal variable selection, PCA was then checked as a dimensionality-reduction step, and two distinct models were estimated for two different purposes."
            ),
            wrap(
                f"The current-year model selected `operating_ratio` and `fiscal_deficit_gdp` and produced a 2024-25 nowcast of {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%. The lagged forecasting model selected `lag_rail_gdp_growth` and `lag_operating_ratio` and produced forecasts of {lagged_horizons.loc[lagged_horizons['horizon']=='short_term_1_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}% for 2024-25, {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}% for the 5-year horizon, and {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}% for the 10-year horizon."
            ),
            wrap(
                "The most important conceptual takeaway is that the two models are not competing versions of the same thing. They are solving different tasks. The current-variable model is a contemporaneous estimate for the current year. The lagged model is the proper forecasting model for future horizons."
            ),
            wrap(
                "If presenting this in class or in the written report, the best final line is: Lasso provided formal variable selection, PCA was evaluated as a dimensionality-reduction step, and the final analysis used a current-variable nowcast for 2024-25 plus a lagged forecasting model for medium-run and long-run railway sector GDP growth."
            ),
        ],
    )


md_lines = [
    "# Professor Workflow Detailed Report",
    "",
    f"- Current-year nowcast for 2024-25: {current_prediction.loc[0, 'predicted_real_gdp_growth_of_railways']:.2f}%",
    f"- Lagged 2024-25 forecast: {lagged_horizons.loc[lagged_horizons['horizon']=='short_term_1_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%",
    f"- Lagged 5-year forecast: {lagged_horizons.loc[lagged_horizons['horizon']=='medium_term_5_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%",
    f"- Lagged 10-year forecast: {lagged_horizons.loc[lagged_horizons['horizon']=='long_run_10_year', 'predicted_real_gdp_growth_of_railways'].iloc[0]:.2f}%",
    "",
    f"- Current Lasso-selected variables: {', '.join(selected_current)}",
    f"- Lagged Lasso-selected variables: {', '.join(selected_lagged)}",
]
MD_PATH.write_text("\n".join(md_lines))

print(PDF_PATH)
