args <- commandArgs(trailingOnly = TRUE)

repoRoot <- if (length(args) >= 1) {
  normalizePath(args[1], mustWork = FALSE)
} else {
  normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
}

dataPath <- file.path(repoRoot, "data", "processed", "railways_master.csv")
outputDir <- file.path(repoRoot, "outputs")
docsDir <- file.path(repoRoot, "docs")

dir.create(outputDir, recursive = TRUE, showWarnings = FALSE)
dir.create(docsDir, recursive = TRUE, showWarnings = FALSE)

railways <- read.csv(dataPath, stringsAsFactors = FALSE)

railways$start_year <- as.integer(substr(railways$year_label, 1, 4))
railways <- railways[order(railways$start_year), ]

pct_change <- function(x) {
  c(NA_real_, 100 * diff(x) / head(x, -1))
}

lag_1 <- function(x) {
  c(NA, head(x, -1))
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

railways$investment_growth <- pct_change(railways$total_investment_rs_cr)
railways$goods_earnings_growth <- pct_change(railways$goods_earnings_rs_cr)
railways$freight_growth <- pct_change(railways$total_revenue_traffic_tonnes_originating)
railways$passenger_km_growth <- pct_change(railways$passenger_kilometres_million)

railways$lag_rail_gdp_growth <- lag_1(railways$real_gdp_growth_of_railways)
railways$lag_investment_growth <- lag_1(railways$investment_growth)
railways$lag_goods_earnings_growth <- lag_1(railways$goods_earnings_growth)
railways$lag_freight_growth <- lag_1(railways$freight_growth)
railways$lag_passenger_km_growth <- lag_1(railways$passenger_km_growth)
railways$lag_fiscal_deficit <- lag_1(railways$fiscal_deficit_gdp)
railways$lag_real_interest_rate <- lag_1(railways$real_interest_rate)
railways$lag_gdp_deflator <- lag_1(railways$gdp_deflator_growth_rate_y_y_chg)

edaVars <- c(
  "real_gdp_growth_of_railways",
  "investment_growth",
  "goods_earnings_growth",
  "freight_growth",
  "lag_rail_gdp_growth",
  "lag_investment_growth",
  "lag_goods_earnings_growth",
  "lag_freight_growth",
  "lag_fiscal_deficit",
  "lag_gdp_deflator"
)

edaSummary <- data.frame(
  variable = edaVars,
  non_missing = sapply(railways[edaVars], function(x) sum(!is.na(x))),
  mean = sapply(railways[edaVars], function(x) mean(x, na.rm = TRUE)),
  median = sapply(railways[edaVars], function(x) median(x, na.rm = TRUE)),
  sd = sapply(railways[edaVars], function(x) sd(x, na.rm = TRUE)),
  min = sapply(railways[edaVars], function(x) min(x, na.rm = TRUE)),
  max = sapply(railways[edaVars], function(x) max(x, na.rm = TRUE))
)

write.csv(edaSummary, file.path(outputDir, "eda_summary.csv"), row.names = FALSE)

missingness <- data.frame(
  variable = names(railways),
  missing_count = sapply(railways, function(x) sum(is.na(x)))
)
write.csv(missingness, file.path(outputDir, "missingness_summary.csv"), row.names = FALSE)

png(file.path(outputDir, "eda_timeseries.png"), width = 1400, height = 1000, res = 160)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(
  railways$start_year,
  railways$real_gdp_growth_of_railways,
  type = "o",
  pch = 16,
  col = "steelblue4",
  xlab = "Start Year",
  ylab = "Percent",
  main = "Railways Real GDP Growth"
)
abline(h = mean(railways$real_gdp_growth_of_railways, na.rm = TRUE), col = "gray50", lty = 2)

plot(
  railways$start_year,
  railways$freight_growth,
  type = "o",
  pch = 16,
  col = "darkgreen",
  xlab = "Start Year",
  ylab = "Percent",
  main = "Freight Growth"
)

plot(
  railways$start_year,
  railways$investment_growth,
  type = "o",
  pch = 16,
  col = "firebrick",
  xlab = "Start Year",
  ylab = "Percent",
  main = "Investment Growth"
)

plot(
  railways$lag_rail_gdp_growth,
  railways$real_gdp_growth_of_railways,
  pch = 16,
  col = "purple4",
  xlab = "Lagged Railways GDP Growth",
  ylab = "Current Railways GDP Growth",
  main = "Persistence Check"
)
abline(lm(real_gdp_growth_of_railways ~ lag_rail_gdp_growth, data = railways), col = "gray30", lwd = 2)
dev.off()

analysisVars <- c(
  "start_year",
  "year_label",
  "real_gdp_growth_of_railways",
  "lag_rail_gdp_growth",
  "lag_investment_growth",
  "lag_goods_earnings_growth",
  "lag_freight_growth",
  "lag_fiscal_deficit",
  "lag_gdp_deflator"
)

correlationData <- railways[analysisVars]
correlationData <- correlationData[complete.cases(correlationData), ]
correlationMatrix <- round(cor(correlationData[sapply(correlationData, is.numeric)]), 3)
write.csv(correlationMatrix, file.path(outputDir, "correlation_matrix.csv"))

candidateFormulas <- list(
  baseline = real_gdp_growth_of_railways ~ lag_rail_gdp_growth,
  freight_investment = real_gdp_growth_of_railways ~ lag_rail_gdp_growth + lag_freight_growth + lag_investment_growth,
  commercial_investment = real_gdp_growth_of_railways ~ lag_rail_gdp_growth + lag_goods_earnings_growth + lag_investment_growth,
  demand_mix = real_gdp_growth_of_railways ~ lag_rail_gdp_growth + lag_goods_earnings_growth + lag_freight_growth
)

evaluate_model <- function(formulaName, formulaObj, data, testYears = 3) {
  modelFrame <- model.frame(formulaObj, data = data, na.action = na.omit)
  modelData <- data[as.numeric(rownames(modelFrame)), ]
  modelData <- modelData[order(modelData$start_year), ]

  trainData <- modelData[seq_len(nrow(modelData) - testYears), ]
  testData <- modelData[(nrow(modelData) - testYears + 1):nrow(modelData), ]

  fit <- lm(formulaObj, data = trainData)
  trainPred <- predict(fit, newdata = trainData)
  testPred <- predict(fit, newdata = testData)

  fitSummary <- summary(fit)

  data.frame(
    model = formulaName,
    observations = nrow(modelData),
    train_observations = nrow(trainData),
    test_observations = nrow(testData),
    train_rmse = rmse(trainData$real_gdp_growth_of_railways, trainPred),
    test_rmse = rmse(testData$real_gdp_growth_of_railways, testPred),
    test_mae = mae(testData$real_gdp_growth_of_railways, testPred),
    adjusted_r_squared = fitSummary$adj.r.squared,
    aic = AIC(fit)
  )
}

modelComparison <- do.call(
  rbind,
  lapply(names(candidateFormulas), function(modelName) {
    evaluate_model(modelName, candidateFormulas[[modelName]], railways)
  })
)

modelComparison <- modelComparison[order(modelComparison$test_rmse), ]
write.csv(modelComparison, file.path(outputDir, "model_comparison.csv"), row.names = FALSE)

bestModelName <- modelComparison$model[1]
bestFormula <- candidateFormulas[[bestModelName]]

finalModelData <- railways[as.numeric(rownames(model.frame(bestFormula, data = railways, na.action = na.omit))), ]
finalModelData <- finalModelData[order(finalModelData$start_year), ]

finalModel <- lm(bestFormula, data = finalModelData)
finalSummary <- summary(finalModel)

coefTable <- as.data.frame(finalSummary$coefficients)
coefTable$term <- rownames(coefTable)
rownames(coefTable) <- NULL
names(coefTable) <- c("estimate", "std_error", "t_value", "p_value", "term")
coefTable <- coefTable[, c("term", "estimate", "std_error", "t_value", "p_value")]
write.csv(coefTable, file.path(outputDir, "final_model_coefficients.csv"), row.names = FALSE)

fittedValues <- predict(finalModel, newdata = finalModelData)
predictionFrame <- data.frame(
  year_label = finalModelData$year_label,
  actual = finalModelData$real_gdp_growth_of_railways,
  fitted = fittedValues,
  residual = finalModelData$real_gdp_growth_of_railways - fittedValues
)
write.csv(predictionFrame, file.path(outputDir, "predictions.csv"), row.names = FALSE)

finalMetrics <- data.frame(
  model = bestModelName,
  observations = nrow(finalModelData),
  r_squared = finalSummary$r.squared,
  adjusted_r_squared = finalSummary$adj.r.squared,
  residual_standard_error = finalSummary$sigma,
  in_sample_rmse = rmse(predictionFrame$actual, predictionFrame$fitted),
  in_sample_mae = mae(predictionFrame$actual, predictionFrame$fitted)
)
write.csv(finalMetrics, file.path(outputDir, "final_model_metrics.csv"), row.names = FALSE)

forecastRow <- railways[railways$year_label == "2024-25", ]
forecastRow <- forecastRow[complete.cases(model.frame(delete.response(terms(finalModel)), data = forecastRow, na.action = na.pass)), ]

if (nrow(forecastRow) == 1) {
  forecastValue <- predict(finalModel, newdata = forecastRow)
  forecastOutput <- data.frame(
    forecast_year = forecastRow$year_label,
    predicted_real_gdp_growth_of_railways = as.numeric(forecastValue)
  )
  write.csv(forecastOutput, file.path(outputDir, "forecast_2024_25.csv"), row.names = FALSE)
}

next_year_label <- function(startYear) {
  paste0(startYear, "-", substr(startYear + 1, 3, 4))
}

recentGoodsGrowth <- tail(stats::na.omit(railways$goods_earnings_growth), 3)
recentFreightGrowth <- tail(stats::na.omit(railways$freight_growth), 3)

goodsGrowthBaseline <- mean(recentGoodsGrowth)
freightGrowthBaseline <- mean(recentFreightGrowth)
goodsGrowthSd <- sd(stats::na.omit(railways$goods_earnings_growth))
freightGrowthSd <- sd(stats::na.omit(railways$freight_growth))

scenarioAssumptions <- data.frame(
  scenario = c("baseline", "optimistic", "pessimistic"),
  goods_growth_assumption = c(
    goodsGrowthBaseline,
    goodsGrowthBaseline + 0.5 * goodsGrowthSd,
    goodsGrowthBaseline - 0.5 * goodsGrowthSd
  ),
  freight_growth_assumption = c(
    freightGrowthBaseline,
    freightGrowthBaseline + 0.5 * freightGrowthSd,
    freightGrowthBaseline - 0.5 * freightGrowthSd
  )
)

lastObservedGrowth <- tail(stats::na.omit(railways$real_gdp_growth_of_railways), 1)
scenarioForecasts <- do.call(
  rbind,
  lapply(seq_len(nrow(scenarioAssumptions)), function(i) {
    assumptionRow <- scenarioAssumptions[i, ]
    directForecastRow <- railways[railways$year_label == "2024-25", ]
    directFirstYearForecast <- as.numeric(predict(finalModel, newdata = directForecastRow))
    prevGrowth <- directFirstYearForecast
    out <- vector("list", 10)

    for (h in seq_len(10)) {
      predictedGrowth <- if (h == 1) {
        directFirstYearForecast
      } else {
        coef(finalModel)[1] +
          coef(finalModel)["lag_rail_gdp_growth"] * prevGrowth +
          coef(finalModel)["lag_goods_earnings_growth"] * assumptionRow$goods_growth_assumption +
          coef(finalModel)["lag_freight_growth"] * assumptionRow$freight_growth_assumption
      }

      forecastStartYear <- 2023 + h
      out[[h]] <- data.frame(
        scenario = assumptionRow$scenario,
        horizon_years_ahead = h,
        forecast_year = next_year_label(forecastStartYear),
        predicted_real_gdp_growth_of_railways = as.numeric(predictedGrowth)
      )
      prevGrowth <- predictedGrowth
    }

    do.call(rbind, out)
  })
)

write.csv(scenarioForecasts, file.path(outputDir, "scenario_forecast_path.csv"), row.names = FALSE)

baselineForecasts <- subset(scenarioForecasts, scenario == "baseline")
horizonForecasts <- baselineForecasts[c(1, 3, 5, 10), c("forecast_year", "predicted_real_gdp_growth_of_railways")]
horizonForecasts$horizon <- c(
  "short_term_1_year",
  "medium_term_3_year",
  "medium_term_5_year",
  "long_run_10_year"
)
horizonForecasts <- horizonForecasts[, c("horizon", "forecast_year", "predicted_real_gdp_growth_of_railways")]
write.csv(horizonForecasts, file.path(outputDir, "horizon_forecasts.csv"), row.names = FALSE)

modelSuitability <- data.frame(
  model = c(
    "lagged_linear_regression",
    "ridge_or_lasso",
    "regression_tree_random_forest_boosting",
    "knn",
    "pcr_pls"
  ),
  suitable = c("yes", "maybe", "no", "no", "maybe"),
  reason = c(
    "Best fit for a small annual sample. Interpretable and aligned with the assignment.",
    "Useful only as a robustness check if you try a slightly larger predictor set.",
    "Too little data. Tree-based models would overfit and become unstable.",
    "Weak fit for a short annual macro sample and hard to explain economically.",
    "Could help compress many macro variables, but secondary to lagged OLS here."
  )
)
write.csv(modelSuitability, file.path(outputDir, "model_suitability.csv"), row.names = FALSE)

png(file.path(outputDir, "actual_vs_predicted.png"), width = 1400, height = 900, res = 160)
plot(
  finalModelData$start_year,
  finalModelData$real_gdp_growth_of_railways,
  type = "o",
  pch = 16,
  col = "steelblue4",
  ylim = range(c(finalModelData$real_gdp_growth_of_railways, fittedValues), na.rm = TRUE),
  xlab = "Start Year",
  ylab = "Percent",
  main = sprintf("Actual vs Fitted: %s", bestModelName)
)
lines(
  finalModelData$start_year,
  fittedValues,
  type = "o",
  pch = 17,
  col = "firebrick"
)
legend("topleft", legend = c("Actual", "Fitted"), col = c("steelblue4", "firebrick"), pch = c(16, 17), lty = 1, bty = "n")
dev.off()

write.csv(finalModelData, file.path(outputDir, "model_dataset.csv"), row.names = FALSE)

forecastText <- if (file.exists(file.path(outputDir, "forecast_2024_25.csv"))) {
  forecastFrame <- read.csv(file.path(outputDir, "forecast_2024_25.csv"), stringsAsFactors = FALSE)
  sprintf(
    "Using the selected lagged linear regression, the predicted railways real GDP growth for %s is %.2f%%.",
    forecastFrame$forecast_year[1],
    forecastFrame$predicted_real_gdp_growth_of_railways[1]
  )
} else {
  "A 2024-25 forecast was not generated because at least one lagged predictor was unavailable."
}

reportLines <- c(
  "# Railways Lagged Linear Regression",
  "",
  "## Objective",
  "",
  "Predict India railways sector real GDP growth using only lagged linear regression style features built from the provided railway and macroeconomic dataset.",
  "",
  "## Data and EDA",
  "",
  sprintf("- Source file transformed into `data/processed/railways_master.csv` with %d yearly rows.", nrow(railways)),
  sprintf("- Usable target observations: %d yearly values for `real_gdp_growth_of_railways`.", sum(!is.na(railways$real_gdp_growth_of_railways))),
  "- Growth features were constructed from total investment, goods earnings, freight traffic, and passenger kilometres.",
  "- One-period lags were created so the regression uses past information rather than current-year leakage.",
  "",
  "## Candidate Models",
  "",
  "- `baseline`: lagged railways GDP growth only",
  "- `freight_investment`: lagged railways GDP growth, lagged freight growth, lagged investment growth",
  "- `commercial_investment`: lagged railways GDP growth, lagged goods earnings growth, lagged investment growth",
  "- `demand_mix`: lagged railways GDP growth, lagged goods earnings growth, lagged freight growth",
  "",
  "## Selected Model",
  "",
  sprintf("- Best holdout model: `%s`", bestModelName),
  sprintf("- Adjusted R-squared: %.3f", finalSummary$adj.r.squared),
  sprintf("- In-sample RMSE: %.3f", finalMetrics$in_sample_rmse[1]),
  "",
  "## Forecast Horizons",
  "",
  sprintf("- Short term (1 year, %s): %.2f%%", horizonForecasts$forecast_year[1], horizonForecasts$predicted_real_gdp_growth_of_railways[1]),
  sprintf("- Medium term (3 years, %s): %.2f%%", horizonForecasts$forecast_year[2], horizonForecasts$predicted_real_gdp_growth_of_railways[2]),
  sprintf("- Medium term (5 years, %s): %.2f%%", horizonForecasts$forecast_year[3], horizonForecasts$predicted_real_gdp_growth_of_railways[3]),
  sprintf("- Long run (10 years, %s): %.2f%%", horizonForecasts$forecast_year[4], horizonForecasts$predicted_real_gdp_growth_of_railways[4]),
  "",
  "## Interpretation",
  "",
  "The lagged setup keeps the exercise aligned with the coursework requirement while making the prediction problem economically sensible for annual railways data. Because the sample is small, the final specification remains intentionally parsimonious.",
  "",
  "## Forecast",
  "",
  forecastText,
  "",
  "Longer-horizon forecasts are generated recursively under a baseline assumption that goods-earnings growth and freight growth stay near their recent 3-year averages. Optimistic and pessimistic scenario paths are saved in `outputs/scenario_forecast_path.csv`.",
  "",
  "## Other Models",
  "",
  "- Lagged linear regression should remain the main model.",
  "- Ridge or Lasso can be added as a robustness check if you want one extra class-friendly comparison.",
  "- Tree-based models, KNN, and other flexible machine learning models are not a good idea with this small annual sample.",
  "",
  "## Output Files",
  "",
  "- `outputs/eda_summary.csv`",
  "- `outputs/missingness_summary.csv`",
  "- `outputs/correlation_matrix.csv`",
  "- `outputs/model_comparison.csv`",
  "- `outputs/final_model_coefficients.csv`",
  "- `outputs/final_model_metrics.csv`",
  "- `outputs/predictions.csv`",
  "- `outputs/horizon_forecasts.csv`",
  "- `outputs/scenario_forecast_path.csv`",
  "- `outputs/model_suitability.csv`",
  "- `outputs/actual_vs_predicted.png`",
  "- `outputs/eda_timeseries.png`"
)

writeLines(reportLines, file.path(docsDir, "report.md"))

message(sprintf("Best model: %s", bestModelName))
