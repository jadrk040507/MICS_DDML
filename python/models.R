# ==============================================================================
# MICS DoubleML - Estimation
# ==============================================================================

library(DoubleML)
library(data.table)

# Source configuration and functions
source("code/config.R")
source("code/data.R")
source("code/learners.R")

# ==============================================================================
# ESTIMATION FUNCTIONS
# ==============================================================================

estimate_effect <- function(dt, outcome_var, treatment_var, learner_name, 
                             learner, include_source_ecoli = TRUE,
                             subgroup_var = NULL, subgroup_val = NULL,
                             checkpoint_dir = CHECKPOINT_DIR) {
  # Estimate causal effect using DoubleML IRM
  # Returns: list with coef, se, ci, n, model
  
  # Create checkpoint filename
  checkpoint_name <- function() {
    parts <- c(outcome_var, treatment_var)
    if (!is.null(subgroup_var)) {
      parts <- c(parts, subgroup_var, as.character(subgroup_val))
    }
    parts <- c(parts, learner_name)
    file.path(checkpoint_dir, paste0(paste(parts, collapse = "_"), ".rds"))
  }
  
  # Check for existing checkpoint
  cp_file <- checkpoint_name()
  if (file.exists(cp_file)) {
    cat("  Loading checkpoint:", basename(cp_file), "\n")
    return(readRDS(cp_file))
  }
  
  # Subsample if subgroup
  if (!is.null(subgroup_var) && !is.null(subgroup_val)) {
    dt <- dt[get(subgroup_var) == subgroup_val]
  }
  
  # Create model matrix
  X <- create_model_matrix(dt, include_source_ecoli = include_source_ecoli)
  
  # Filter complete cases
  complete <- complete.cases(X) & 
              !is.na(dt[[outcome_var]]) & 
              !is.na(dt[[treatment_var]])
  
  if (sum(complete) < 50) {
    cat("  ERROR: Too few observations (N =", sum(complete), ")\n")
    return(NULL)
  }
  
  dt_clean <- dt[complete]
  X_clean <- X[complete, ]
  
  # Prepare data for DoubleML
  df <- data.frame(
    y = dt_clean[[outcome_var]],
    d = dt_clean[[treatment_var]],
    X_clean
  )
  
  # Create DoubleML data
  dml_data <- double_ml_data_from_data_frame(
    df, 
    y_col = "y", 
    d_cols = "d", 
    x_cols = colnames(X_clean)
  )
  
  # Fit DoubleML IRM
  dml <- DoubleMLIRM$new(
    data = dml_data,
    ml_g = learner$g,
    ml_m = learner$m,
    n_folds = N_FOLDS,
    n_rep = N_REP,
    score = "ATE"
  )
  
  tryCatch({
    dml$fit()
    
    # Extract results
    result <- list(
      outcome = outcome_var,
      treatment = treatment_var,
      subgroup_var = subgroup_var,
      subgroup_val = subgroup_val,
      learner = learner_name,
      coef = dml$coef,
      se = dml$se,
      ci_lower = dml$confint()[1],
      ci_upper = dml$confint()[2],
      n = sum(complete),
      model = dml
    )
    
    # Save checkpoint
    saveRDS(result, cp_file)
    cat("  Saved:", basename(cp_file), "\n")
    
    return(result)
    
  }, error = function(e) {
    cat("  ERROR:", conditionMessage(e), "\n")
    return(NULL)
  })
}

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

run_analysis <- function(dt, outcomes, treatments, learners,
                          include_source_ecoli = TRUE,
                          subgroups = FALSE,
                          checkpoint_dir = CHECKPOINT_DIR) {
  # Run full analysis
  # Returns: data.table with all results
  
  results <- list()
  idx <- 1
  
  # Get learner list
  learner_list <- if (is.function(learners)) learners() else learners
  
  # Iterate over combinations
  for (out in outcomes) {
    for (treat in treatments) {
      for (ln in names(learner_list)) {
        
        cat("\n", out$label, "|", treat$label, "|", ln, "\n")
        
        res <- estimate_effect(
          dt = dt,
          outcome_var = out$var,
          treatment_var = treat$var,
          learner_name = ln,
          learner = learner_list[[ln]],
          include_source_ecoli = include_source_ecoli,
          checkpoint_dir = checkpoint_dir
        )
        
        if (!is.null(res)) {
          results[[idx]] <- res
          idx <- idx + 1
          cat(sprintf("  Effect: %.4f (%.4f)\n", res$coef, res$se))
        }
      }
    }
  }
  
  # Convert to data.table
  results_dt <- rbindlist(lapply(results, function(r) {
    data.table(
      outcome = r$outcome,
      treatment = r$treatment,
      learner = r$learner,
      coef = r$coef,
      se = r$se,
      ci_lower = r$ci_lower,
      ci_upper = r$ci_upper,
      n = r$n,
      significant = ifelse(r$ci_lower > 0 | r$ci_upper < 0, "***", "")
    )
  }), fill = TRUE)
  
  return(results_dt)
}

# ==============================================================================
# EXPORT RESULTS
# ==============================================================================

export_results <- function(results, filename = "results.rds") {
  # Save results to RDS
  filepath <- file.path(OUTPUT_DIR, filename)
  saveRDS(results, filepath)
  cat("Results saved to:", filepath, "\n")
}

export_latex <- function(results, filename = "tables.tex") {
  # Export results to LaTeX table
  filepath <- file.path(OUTPUT_DIR, filename)
  
  # Format for LaTeX
  latex <- results[, .(
    Outcome = outcome,
    Treatment = treatment,
    Learner = learner,
    Effect = sprintf("%.4f", coef),
    SE = sprintf("%.4f", se),
    CI = sprintf("[%.4f, %.4f]", ci_lower, ci_upper),
    N = n,
    Sig = significant
  )]
  
  # Write LaTeX table
  sink(filepath)
  cat("\\begin{table}[htbp]\n")
  cat("\\centering\n")
  cat("\\caption{DoubleML Estimation Results}\n")
  cat("\\label{tab:results}\n")
  cat("\\begin{tabular}{lccccl}\n")
  cat("\\hline\n")
  cat("Outcome & Treatment & Learner & Effect & SE & CI \\\\\n")
  cat("\\hline\n")
  
  for (i in 1:nrow(latex)) {
    cat(sprintf("%s & %s & %s & %s & %s & %s %s \\\\\n",
                latex$Outcome[i],
                latex$Treatment[i],
                latex$Learner[i],
                latex$Effect[i],
                latex$SE[i],
                latex$CI[i],
                latex$Sig[i]))
  }
  
  cat("\\hline\n")
  cat("\\end{tabular}\n")
  cat("\\end{table}\n")
  sink()
  
  cat("LaTeX table saved to:", filepath, "\n")
}