# ==============================================================================
# MICS DoubleML - Re-run Subgroup Analysis (bugfix)
# ==============================================================================
# Bug: checkpoints didn't include RiskSource in filename, so risk_source=1,2 
# loaded the risk_source=0 checkpoint. Fixed by passing subgroup_var/val through.
# This script reruns ONLY the subgroup analysis.
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
dt_multi[, treat_chlorine := ifelse(chlorine == 1, 1L, 0L)]
dt_multi[, treat_filter := ifelse(filter == 1, 1L, 0L)]
dt_multi[, treat_other := ifelse(other_treat == 1, 1L, 0L)]

# Create learners — OLS excluded from stacked (quasi-complete separation)
# Same approach as multi-treatment in run.R
learners <- create_learners(type = "binary")
learners$stacked <- create_stacked_ensemble_no_ols()

cat("\n========================================\n")
cat("SUBGROUP ANALYSIS BY SOURCE RISK (bugfix)\n")
cat("========================================\n")

# Delete old subgroup checkpoints (they are all duplicates of risk_source=0)
# First, let's identify which checkpoints are subgroup-specific and should be deleted
# The old ones don't have RiskSource in the name, the new ones will
cat("\nNote: Old checkpoints without RiskSource in filename will be ignored\n")
cat("New checkpoints will include RiskSource in filename\n\n")

subgroup_results_binary <- list()
subgroup_results_ecoli <- list()
idx_binary <- 1
idx_ecoli <- 1

for (risk_level in 0:2) {
  cat("\n--- Source Risk Level:", risk_level, "---\n")
  
  dt_sub <- dt_multi[as.integer(RiskSource) == risk_level]
  cat("  Subsample size:", nrow(dt_sub), "\n")
  
  # Binary outcome (diarrhea)
  res_binary <- run_analysis(
    dt = dt_sub,
    outcomes = BINARY_OUTCOMES,
    treatments = list(list(var = "treat_boil", label = "Boil")),
    learners = learners,
    include_source_ecoli = TRUE,
    subgroup_var = "RiskSource",
    subgroup_val = risk_level,
    checkpoint_dir = CHECKPOINT_DIR
  )
  
  if (!is.null(res_binary) && nrow(res_binary) > 0) {
    res_binary[, risk_source := risk_level]
    subgroup_results_binary[[idx_binary]] <- res_binary
    idx_binary <- idx_binary + 1
  }
  
  # E.coli risk outcomes
  res_ecoli <- run_analysis(
    dt = dt_sub,
    outcomes = ECOLI_RISK_OUTCOMES,
    treatments = list(list(var = "treat_boil", label = "Boil")),
    learners = learners,
    include_source_ecoli = FALSE,
    subgroup_var = "RiskSource",
    subgroup_val = risk_level,
    checkpoint_dir = CHECKPOINT_DIR
  )
  
  if (!is.null(res_ecoli) && nrow(res_ecoli) > 0) {
    res_ecoli[, risk_source := risk_level]
    subgroup_results_ecoli[[idx_ecoli]] <- res_ecoli
    idx_ecoli <- idx_ecoli + 1
  }
}

# Combine subgroup results
results_subgroups_binary <- rbindlist(subgroup_results_binary, fill = TRUE)
results_subgroups_binary[, outcome_type := "binary"]

results_subgroups_ecoli <- rbindlist(subgroup_results_ecoli, fill = TRUE)
results_subgroups_ecoli[, outcome_type := "ecoli_risk"]

results_subgroups <- rbindlist(list(
  results_subgroups_binary,
  results_subgroups_ecoli
), fill = TRUE)

# Save
saveRDS(results_subgroups, file.path(OUTPUT_DIR, "results_subgroups.rds"))
cat("\nSubgroup results saved to:", file.path(OUTPUT_DIR, "results_subgroups.rds"), "\n")

# Extract stacked weights for subgroups and merge into stacked_weights.rds
cat("\nExtracting stacked weights from new subgroup checkpoints...\n")
all_weights <- readRDS(file.path(OUTPUT_DIR, "stacked_weights.rds"))

for (risk_level in 0:2) {
  for (outcome_var in c("diarrhea", "some_risk_home", "very_high_risk_home")) {
    key <- paste0(outcome_var, "_RiskSource", risk_level, "_treat_boil")
    cp_file <- file.path(CHECKPOINT_DIR, paste0(key, "_stacked.rds"))
    
    if (file.exists(cp_file)) {
      obj <- readRDS(cp_file)
      if (!is.null(obj$stacked_weights)) {
        weight_key <- paste0(outcome_var, "_RiskSource", risk_level, "_treat_boil")
        all_weights[[weight_key]] <- obj$stacked_weights
        cat("  Added weights for:", weight_key, "\n")
      }
    }
  }
}

saveRDS(all_weights, file.path(OUTPUT_DIR, "stacked_weights.rds"))
cat("Stacked weights (with subgroups) saved.\n")

cat("\n=== SUBGROUP ANALYSIS COMPLETE ===\n")