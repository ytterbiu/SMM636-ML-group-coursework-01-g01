# ============================================================
# RANDOM FOREST (ranger) + 5-fold CV tuning + Metrics:
# AUC, Sensitivity, Specificity, NPV
# FIX: CV sens/spec/npv computed by deriving class from probs
# ============================================================

# ---- 0) Packages ----
pkgs <- c("readr", "dplyr", "tidymodels", "ranger", "ggplot2", "tibble")
to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)

library(readr)
library(dplyr)
library(tidymodels)
library(ranger)
library(ggplot2)
library(tibble)

set.seed(123)

# ---- 1) Read CSV robustly ----
file_path <- "processed.cleveland.data.csv"

df1 <- suppressWarnings(readr::read_csv(file_path, show_col_types = FALSE))

looks_like_numbers <- function(x) mean(grepl("^\\s*-?\\d+(\\.\\d+)?\\s*$", x)) > 0.5

if (ncol(df1) > 0 && looks_like_numbers(names(df1))) {
  message("Detected first row was used as column names. Re-reading with col_names = FALSE...")
  df_raw <- suppressWarnings(readr::read_csv(file_path, col_names = FALSE, show_col_types = FALSE))
} else {
  df_raw <- df1
}

if (ncol(df_raw) <= 1) {
  message("Delimiter/header issue suspected. Re-reading with base::read.csv(header=FALSE)...")
  df_raw <- read.csv(file_path, header = FALSE, sep = ",", stringsAsFactors = FALSE)
  df_raw <- as_tibble(df_raw)
}

# ---- 2) Apply Cleveland names if 14 columns ----
uci_names_14 <- c("age","sex","cp","trestbps","chol","fbs","restecg",
                  "thalach","exang","oldpeak","slope","ca","thal","num")

if (ncol(df_raw) == 14) {
  names(df_raw) <- uci_names_14
} else {
  names(df_raw) <- paste0("V", seq_len(ncol(df_raw)))
  names(df_raw)[ncol(df_raw)] <- "num"
  message("Dataset is not 14 columns. Using LAST column as outcome named 'num'.")
}

# ---- 3) Clean: '?' -> NA and numeric conversion ----
df_clean <- df_raw %>%
  mutate(across(everything(), ~ ifelse(.x == "?", NA, .x))) %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(.x))))

if (!("num" %in% names(df_clean))) stop("Outcome column 'num' not found.")

# ---- 4) Binary target ----
df <- df_clean %>%
  filter(!is.na(num)) %>%
  mutate(
    target = if_else(num == 0, "NoDisease", "Disease"),
    target = factor(target, levels = c("NoDisease", "Disease"))
  ) %>%
  select(-num)

# Drop rows where ALL predictors are NA
predictor_cols <- setdiff(names(df), "target")
df <- df %>%
  filter(rowSums(is.na(across(all_of(predictor_cols)))) < length(predictor_cols))

# ---- 5) Train/test split ----
split_obj <- initial_split(df, prop = 0.8, strata = target)
train_df <- training(split_obj)
test_df  <- testing(split_obj)

# ---- 6) 5-fold CV ----
folds <- vfold_cv(train_df, v = 5, strata = target)

# ---- 7) Recipe ----
rf_recipe <- recipe(target ~ ., data = train_df) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# ---- 8) RF model spec ----
rf_spec <- rand_forest(
  mode  = "classification",
  trees = 1000,
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", probability = TRUE, importance = "impurity")

rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_spec)

# ---- 9) Tuning grid ----
num_pred <- ncol(train_df) - 1
if (num_pred < 1) stop("No predictors found. Check your file.")

mtry_vals  <- unique(pmax(1, round(seq(1, num_pred, length.out = min(6, num_pred)))))
min_n_vals <- c(2, 5, 10, 15, 20)

grid <- expand_grid(
  mtry  = mtry_vals,
  min_n = min_n_vals
)

# ---- 10) Tune by AUC ----
cv_metrics <- metric_set(yardstick::roc_auc)
ctrl <- control_grid(save_pred = TRUE, verbose = TRUE)

rf_tuned <- tune_grid(
  rf_wf,
  resamples = folds,
  grid      = grid,
  metrics   = cv_metrics,
  control   = ctrl
)

cat("\n==============================\n")
cat("CROSS-VALIDATION RESULTS (AUC)\n")
cat("==============================\n")
print(collect_metrics(rf_tuned))

best_by_auc <- select_best(rf_tuned, metric = "roc_auc")

cat("\n==============================\n")
cat("BEST HYPERPARAMETERS (BY AUC)\n")
cat("==============================\n")
print(best_by_auc)

# ---- 11) Fit final tuned model ----
final_wf  <- finalize_workflow(rf_wf, best_by_auc)
final_fit <- fit(final_wf, data = train_df)

# ---- 12) TEST set predictions ----
test_pred <- predict(final_fit, test_df, type = "prob") %>%
  bind_cols(predict(final_fit, test_df, type = "class")) %>%
  bind_cols(test_df %>% select(target)) %>%
  rename(pred_class = .pred_class)

if (!(".pred_Disease" %in% names(test_pred))) {
  stop("Missing '.pred_Disease' probability column. Check target levels.")
}

# ---- 13) TEST metrics ----
test_auc  <- yardstick::roc_auc(test_pred, truth = target, .pred_Disease, event_level = "second")
test_sens <- yardstick::sens(test_pred, truth = target, estimate = pred_class, event_level = "second")
test_spec <- yardstick::spec(test_pred, truth = target, estimate = pred_class, event_level = "second")
test_npv  <- yardstick::npv(test_pred,  truth = target, estimate = pred_class, event_level = "second")

test_metrics_tbl <- bind_rows(test_auc, test_sens, test_spec, test_npv) %>%
  mutate(dataset = "Test Set") %>%
  select(dataset, .metric, .estimate)

cat("\n==============================\n")
cat("FINAL TUNED RANDOM FOREST - TEST SET METRICS\n")
cat("AUC, Sensitivity, Specificity, NPV\n")
cat("==============================\n")
print(test_metrics_tbl)

cat("\n==============================\n")
cat("CONFUSION MATRIX (TEST SET)\n")
cat("==============================\n")
print(conf_mat(test_pred, truth = target, estimate = pred_class))

# ---- 14) CV metrics for BEST model (mean +/- sd) ----
# collect_predictions only stores probs; derive class from prob threshold
threshold <- 0.5

cv_best_preds <- collect_predictions(rf_tuned, parameters = best_by_auc)

if (!(".pred_Disease" %in% names(cv_best_preds))) {
  stop("CV predictions do not contain '.pred_Disease'. Check class levels / probability columns.")
}

cv_best_preds <- cv_best_preds %>%
  mutate(
    pred_class = factor(
      if_else(.pred_Disease >= threshold, "Disease", "NoDisease"),
      levels = levels(target)
    )
  )

# per-fold metrics
cv_by_fold <- cv_best_preds %>%
  group_by(id) %>%
  summarise(
    auc  = yardstick::roc_auc_vec(truth = target, estimate = .pred_Disease, event_level = "second"),
    sens = yardstick::sens_vec(truth = target, estimate = pred_class, event_level = "second"),
    spec = yardstick::spec_vec(truth = target, estimate = pred_class, event_level = "second"),
    npv  = yardstick::npv_vec(truth = target, estimate = pred_class, event_level = "second"),
    .groups = "drop"
  )

cv_summary <- tibble(
  dataset = "5-fold CV (best AUC settings)",
  .metric = c("roc_auc", "sens", "spec", "npv"),
  mean    = c(mean(cv_by_fold$auc), mean(cv_by_fold$sens), mean(cv_by_fold$spec), mean(cv_by_fold$npv)),
  sd      = c(sd(cv_by_fold$auc),   sd(cv_by_fold$sens),   sd(cv_by_fold$spec),   sd(cv_by_fold$npv))
)

cat("\n==============================\n")
cat("CROSS-VALIDATION METRICS (BEST SETTINGS)\n")
cat("mean +/- sd across folds (threshold = 0.5 for class metrics)\n")
cat("==============================\n")
print(cv_summary)

# ---- 15) ROC curve plot (TEST SET) ----
roc_df <- yardstick::roc_curve(test_pred, truth = target, .pred_Disease, event_level = "second")

roc_plot <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(linetype = 2) +
  labs(
    title = "ROC Curve (Test Set) - Random Forest",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  )

out_file <- file.path(getwd(), "roc_curve_random_forest.png")
ggsave(filename = out_file, plot = roc_plot, width = 6, height = 5, dpi = 300)
cat("\nROC plot saved to:\n", out_file, "\n")

# ---- 16) Export metrics ----
cv_export <- cv_summary %>%
  transmute(dataset, .metric, .estimate = mean, sd)

test_export <- test_metrics_tbl %>%
  mutate(sd = NA_real_)

results_export <- bind_rows(test_export, cv_export)

write.csv(results_export, "random_forest_metrics_results.csv", row.names = FALSE)
cat("\nMetrics saved to:\n", file.path(getwd(), "random_forest_metrics_results.csv"), "\n")