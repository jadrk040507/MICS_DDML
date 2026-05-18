# ==============================================================================
# MICS DoubleML - Main Analysis Script
# ==============================================================================
# Run DoubleML estimation for water treatment effects on diarrhea and E.coli
# 
# Usage:
#   source("code/config.R")
#   source("code/run.R")
# ==============================================================================

library(here)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(haven)
library(data.table)

# Source modules
source(here::here("code", "config.R"))
source(here::here("code", "data.R"))
source(here::here("code", "learners.R"))
source(here::here("code", "models.R"))

# ==============================================================================
# 1. LOAD AND PREPARE DATA
# ==============================================================================

cat("\n========================================\n")
cat("MICS DoubleML Analysis\n")
cat("========================================\n\n")

cat("Loading data...\n")
dt <- prepare_data(DATA_FILE)
get_summary(dt)

# ==============================================================================
# 2. ANALYSIS 1: ANY TREATMENT (Binary)
# ==============================================================================

cat("\n========================================\n")
cat("ANALYSIS 1: ANY WATER TREATMENT\n")
cat("========================================\n")

# Create treatment variable (binary: any treatment vs no treatment)
dt_analysis <- dt[!is.na(any_treatment)]

# Create learners
learners <- create_learners(type = "binary")

# Add stacked ensemble
learners$stacked <- create_stacked_ensemble()

# Run analysis for binary outcome (diarrhea)
results_any_binary <- run_analysis(
  dt = dt_analysis,
  outcomes = BINARY_OUTCOMES,
  treatments = list(ANY_TREATMENT),
  learners = learners,
  include_source_ecoli = TRUE,  # Include log_WQ27 for diarrhea model
  checkpoint_dir = CHECKPOINT_DIR
)

# Run analysis for continuous outcome (log_WQ26)
results_any_continuous <- run_analysis(
  dt = dt_analysis,
  outcomes = CONTINUOUS_OUTCOMES,
  treatments = list(ANY_TREATMENT),
  learners = learners,
  include_source_ecoli = FALSE,  # NO log_WQ27 for E.coli model
  checkpoint_dir = CHECKPOINT_DIR
)

# Combine results
results_any <- rbindlist(list(
  results_any_binary[, outcome_type := "binary"],
  results_any_continuous[, outcome_type := "continuous"]
), fill = TRUE)

# Save results
export_results(results_any, "results_any_treatment.rds")

# ==============================================================================
# 3. ANALYSIS 2: MULTI TREATMENT (Specific methods vs control)
# ==============================================================================

cat("\n========================================\n")
cat("ANALYSIS 2: SPECIFIC TREATMENT METHODS\n")
cat("========================================\n")

# Create treatment indicators (vs control: no treatment)
# Note: WQ15_g is mutually exclusive - each household reports ONE primary method
# WQ15_g = 0 (no treatment), 1 (boil), 2 (chlorine), 3 (filter), 98 (other)
dt_multi <- dt[no_treatment == 1 | boil == 1 | chlorine == 1 | filter == 1 | other_treat == 1]
dt_multi[, treat_boil := ifelse(boil == 1, 1L, 0L)]
dt_multi[, treat_chlorine := ifelse(chlorine == 1, 1L, 0L)]
dt_multi[, treat_filter := ifelse(filter == 1, 1L, 0L)]
dt_multi[, treat_other := ifelse(other_treat == 1, 1L, 0L)]

# Define treatments for multi analysis
multi_treatments <- list(
  list(var = "treat_boil", label = "Boil"),
  list(var = "treat_chlorine", label = "Chlorine"),
  list(var = "treat_filter", label = "Filter"),
  list(var = "treat_other", label = "Other")
)

# Run analysis for binary outcome (diarrhea)
results_multi_binary <- run_analysis(
  dt = dt_multi,
  outcomes = BINARY_OUTCOMES,
  treatments = multi_treatments,
  learners = learners,
  include_source_ecoli = TRUE,
  checkpoint_dir = CHECKPOINT_DIR
)

# Run analysis for continuous outcome (log_WQ26)
results_multi_continuous <- run_analysis(
  dt = dt_multi,
  outcomes = CONTINUOUS_OUTCOMES,
  treatments = multi_treatments,
  learners = learners,
  include_source_ecoli = FALSE,  # NO log_WQ27 for E.coli model
  checkpoint_dir = CHECKPOINT_DIR
)

# Combine results
results_multi <- rbindlist(list(
  results_multi_binary[, outcome_type := "binary"],
  results_multi_continuous[, outcome_type := "continuous"]
), fill = TRUE)

# Save results
export_results(results_multi, "results_multi_treatment.rds")

# ==============================================================================
# 4. ANALYSIS 3: SUBGROUPS BY SOURCE RISK
# ==============================================================================

cat("\n========================================\n")
cat("ANALYSIS 3: SUBGROUPS BY SOURCE RISK\n")
cat("========================================\n")

# Subgroup analysis for Boil (most effective treatment)
# Run for both binary and continuous outcomes

subgroup_results_binary <- list()
subgroup_results_continuous <- list()
idx_binary <- 1
idx_continuous <- 1

for (risk_level in 0:2) {
  cat("\n--- Source Risk Level:", risk_level, "---\n")
  
  dt_sub <- dt_multi[risk_source == risk_level]
  
  # Binary outcome (diarrhea)
  res_binary <- run_analysis(
    dt = dt_sub,
    outcomes = BINARY_OUTCOMES,
    treatments = list(list(var = "treat_boil", label = "Boil")),
    learners = learners,
    include_source_ecoli = TRUE,
    subgroup_var = "risk_source",
    subgroup_val = risk_level,
    checkpoint_dir = CHECKPOINT_DIR
  )
  
  if (!is.null(res_binary) && nrow(res_binary) > 0) {
    res_binary[, risk_source := risk_level]
    subgroup_results_binary[[idx_binary]] <- res_binary
    idx_binary <- idx_binary + 1
  }
  
  # Continuous outcome (log_WQ26)
  res_continuous <- run_analysis(
    dt = dt_sub,
    outcomes = CONTINUOUS_OUTCOMES,
    treatments = list(list(var = "treat_boil", label = "Boil")),
    learners = learners,
    include_source_ecoli = FALSE,  # NO log_WQ27 for E.coli model
    subgroup_var = "risk_source",
    subgroup_val = risk_level,
    checkpoint_dir = CHECKPOINT_DIR
  )
  
  if (!is.null(res_continuous) && nrow(res_continuous) > 0) {
    res_continuous[, risk_source := risk_level]
    subgroup_results_continuous[[idx_continuous]] <- res_continuous
    idx_continuous <- idx_continuous + 1
  }
}

# Combine subgroup results
results_subgroups_binary <- rbindlist(subgroup_results_binary, fill = TRUE)
results_subgroups_binary[, outcome_type := "binary"]

results_subgroups_continuous <- rbindlist(subgroup_results_continuous, fill = TRUE)
results_subgroups_continuous[, outcome_type := "continuous"]

results_subgroups <- rbindlist(list(
  results_subgroups_binary,
  results_subgroups_continuous
), fill = TRUE)

export_results(results_subgroups, "results_subgroups.rds")

# ==============================================================================
# 5. EXPORT RESULTS
# ==============================================================================

cat("\n========================================\n")
cat("EXPORTING RESULTS\n")
cat("========================================\n")

# Combine all results
all_results <- rbindlist(list(
  results_any[, analysis := "any_treatment"],
  results_multi[, analysis := "multi_treatment"],
  results_subgroups[, analysis := "subgroups"]
), fill = TRUE)

# Export LaTeX tables
export_latex(all_results, "tables.tex")

# Save combined results
export_results(all_results, "results_all.rds")

# Summary
cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to:", OUTPUT_DIR, "\n")
cat("Checkpoints saved to:", CHECKPOINT_DIR, "\n")

# Print summary
cat("\n=== SUMMARY OF SIGNIFICANT EFFECTS ===\n\n")

# Any treatment
sig_any <- all_results[analysis == "any_treatment" & significant == "***"]
if (nrow(sig_any) > 0) {
  cat("Any Treatment:\n")
  print(sig_any[, .(outcome, treatment, learner, coef, se, n)])
}

# Multi treatment
sig_multi <- all_results[analysis == "multi_treatment" & significant == "***"]
if (nrow(sig_multi) > 0) {
  cat("\nSpecific Treatments:\n")
  print(sig_multi[, .(outcome, treatment, learner, coef, se, n)])
}

# Subgroups
sig_sub <- all_results[analysis == "subgroups" & significant == "***"]
if (nrow(sig_sub) > 0) {
  cat("\nSubgroups (by Source Risk):\n")
  print(sig_sub[, .(outcome, treatment, risk_source, learner, coef, se, n)])
}