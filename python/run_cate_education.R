# ==============================================================================
# MICS DoubleML - CATE by Education Level
# ==============================================================================
# Conditional Average Treatment Effects for some_risk_home and 
# very_high_risk_home, subgrouped by household head education level.
# Education levels: 0=None, 1=Primary, 2=Lower secondary, 
#                   3=Upper secondary, 4=College/higher
# Treatment: treat_boil only
# Learners: ols, lasso, ridge, enet, rf, xgb + stacked (NO OLS in stacked)
# ==============================================================================

library(here)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(haven)
library(data.table)

source(here::here("code", "config.R"))
source(here::here("code", "data.R"))
source(here::here("code", "learners.R"))
source(here::here("code", "models.R"))

# Load data
cat("Loading data...\n")
dt <- prepare_data(DATA_FILE)

# Prepare multi-treatment data (same as run.R)
dt_multi <- dt[no_treatment == 1 | boil == 1 | chlorine == 1 | filter == 1 | other_treat == 1]
dt_multi[, treat_boil := ifelse(boil == 1, 1L, 0L)]

# Education levels to loop over
edu_levels <- 0:4
edu_labels <- c("None", "Primary", "LowerSec", "UpperSec", "College")

# Create learners — OLS excluded from stacked (quasi-complete separation)
learners <- create_learners(type = "binary")
learners$stacked <- create_stacked_ensemble_no_ols()

cat("\n========================================\n")
cat("CATE BY EDUCATION LEVEL\n")
cat("========================================\n")

cate_results <- list()
idx <- 1

for (i in seq_along(edu_levels)) {
  edu_val <- edu_levels[i]
  edu_lbl <- edu_labels[i]
  
  cat(sprintf("\n--- Education Level: %s (helevel = %d) ---\n", edu_lbl, edu_val))
  
  dt_sub <- dt_multi[!is.na(helevel) & as.integer(helevel) == edu_val]
  cat("  Subsample size:", nrow(dt_sub), "\n")
  
  if (nrow(dt_sub) < 100) {
    cat("  SKIP: Too few observations\n")
    next
  }
  
  # E.coli risk outcomes (NO source ecoli — it's a mediator)
  res_ecoli <- run_analysis(
    dt = dt_sub,
    outcomes = ECOLI_RISK_OUTCOMES,
    treatments = list(list(var = "treat_boil", label = "Boil")),
    learners = learners,
    include_source_ecoli = FALSE,
    subgroup_var = "helevel",
    subgroup_val = edu_val,
    checkpoint_dir = CHECKPOINT_DIR
  )
  
  if (!is.null(res_ecoli) && nrow(res_ecoli) > 0) {
    res_ecoli[, education_level := edu_val]
    res_ecoli[, education_label := edu_lbl]
    cate_results[[idx]] <- res_ecoli
    idx <- idx + 1
  }
}

# Combine results
results_cate <- rbindlist(cate_results, fill = TRUE)

# Save
saveRDS(results_cate, file.path(OUTPUT_DIR, "results_cate_education.rds"))
cat("\nCATE (Education) results saved to:", file.path(OUTPUT_DIR, "results_cate_education.rds"), "\n")

# Print summary
cat("\n=== CATE BY EDUCATION - SUMMARY ===\n\n")
if (nrow(results_cate) > 0) {
  # Focus on stacked learner for summary
  stacked_results <- results_cate[learner == "stacked"]
  if (nrow(stacked_results) > 0) {
    cat("Stacked Ensemble Results:\n")
    cat(sprintf("%-15s %-20s %-8s %10s %10s %8s\n", 
                "Education", "Outcome", "N", "Effect", "SE", "Sig"))
    cat(strrep("-", 75), "\n")
    for (i in 1:nrow(stacked_results)) {
      r <- stacked_results[i]
      cat(sprintf("%-15s %-20s %-8d %10.4f %10.4f %8s\n",
                  r$education_label, r$outcome, r$n, r$coef, r$se, r$significant))
    }
  }
  
  cat("\n\nAll Learners:\n")
  cat(sprintf("%-15s %-20s %-8s %10s %10s %8s\n", 
              "Education", "Outcome", "Learner", "Effect", "SE", "Sig"))
  cat(strrep("-", 80), "\n")
  for (i in 1:nrow(results_cate)) {
    r <- results_cate[i]
    cat(sprintf("%-15s %-20s %-8s %10.4f %10.4f %8s\n",
                r$education_label, r$outcome, r$learner, r$coef, r$se, r$significant))
  }
}

cat("\n=== CATE BY EDUCATION COMPLETE ===\n")