# ==============================================================================
# MICS DoubleML - Data Preparation
# ==============================================================================

library(haven)
library(data.table)

prepare_data <- function(filepath = DATA_FILE) {
  # Load data
  dt <- as.data.table(read_dta(filepath))
  
  # ==============================================================================
  # OUTCOMES
  # ==============================================================================
  
  # Binary: Diarrhea (children under 5)
  dt[, diarrhea := as.integer(HH_child_diarrhea)]
  
  # Continuous: log(1 + E.coli count) at home
  dt[, log_WQ26 := log1p(WQ26)]  # E.coli at home
  dt[, log_WQ27 := log1p(WQ27)]  # E.coli at source (confounder for diarrhea)
  
  # ==============================================================================
  # TREATMENTS
  # ==============================================================================
  
  # Any water treatment (includes boil, chlorine, filter, other)
  dt[, any_treatment := as.integer(water_treatment)]
  
  # Specific treatments (vs control = no treatment)
  # Note: WQ15_g = 0 (no treatment), 1 (boil), 2 (chlorine), 3 (filter), 98 (other)
  dt[, no_treatment := as.integer(WQ15_g_0)]
  dt[, boil := as.integer(WQ15_g_1)]
  dt[, chlorine := as.integer(WQ15_g_2)]
  dt[, filter := as.integer(WQ15_g_3)]
  dt[, other_treat := as.integer(WQ15_g_98)]  # Other treatment
  
  # ==============================================================================
  # CONFOUNDERS
  # ==============================================================================
  
  # Wealth quintile dummies
  dt[, wealth_q1 := ifelse(as.integer(windex5) == 1, 1L, 0L)]
  dt[, wealth_q2 := ifelse(as.integer(windex5) == 2, 1L, 0L)]
  dt[, wealth_q3 := ifelse(as.integer(windex5) == 3, 1L, 0L)]
  dt[, wealth_q4 := ifelse(as.integer(windex5) == 4, 1L, 0L)]
  dt[, wealth_q5 := ifelse(as.integer(windex5) == 5, 1L, 0L)]
  
  # Education dummies
  dt[, edu_0 := ifelse(is.na(helevel), 0L, ifelse(as.integer(helevel) == 0, 1L, 0L))]
  dt[, edu_1 := ifelse(is.na(helevel), 0L, ifelse(as.integer(helevel) == 1, 1L, 0L))]
  dt[, edu_2 := ifelse(is.na(helevel), 0L, ifelse(as.integer(helevel) == 2, 1L, 0L))]
  dt[, edu_3 := ifelse(is.na(helevel), 0L, ifelse(as.integer(helevel) == 3, 1L, 0L))]
  dt[, edu_4 := ifelse(is.na(helevel), 0L, ifelse(as.integer(helevel) == 4, 1L, 0L))]
  dt[, edu_na := ifelse(is.na(helevel), 1L, 0L)]
  
  # Urban
  dt[, urban_bin := as.integer(urban)]
  
  # Sanitation
  dt[, sanitation := as.integer(improved_latrine)]
  
  # Number of children
  dt[, num_children := as.integer(HHCHILDREN)]
  dt[is.na(num_children), num_children := 0L]
  
  # Water source (factor for dummies)
  dt[, water_source := as.factor(WS1)]
  
  # Country (factor for FE)
  dt[, country := as.factor(Country)]
  
  # ==============================================================================
  # SUBGROUPS
  # ==============================================================================
  
  # Source risk (0 = no risk, 1 = moderate, 2 = high)
  dt[, risk_source := as.integer(RiskSource)]
  
  return(dt)
}

# ==============================================================================
# CREATE MODEL MATRIX
# ==============================================================================

create_model_matrix <- function(dt, include_source_ecoli = TRUE) {
  # Base confounders
  X_base <- c(
    "wealth_q1", "wealth_q2", "wealth_q3", "wealth_q4", "wealth_q5",
    "edu_0", "edu_1", "edu_2", "edu_3", "edu_4", "edu_na",
    "urban_bin", "sanitation", "num_children"
  )
  
  # Water source dummies
  ws_mm <- model.matrix(~ water_source - 1, data = dt)
  colnames(ws_mm) <- gsub("water_source", "ws_", colnames(ws_mm))
  
  # Country dummies
  country_mm <- model.matrix(~ country - 1, data = dt)
  colnames(country_mm) <- gsub(" ", "_", colnames(country_mm))
  colnames(country_mm) <- gsub("country", "ctry_", colnames(country_mm))
  
  # Combine
  X <- as.matrix(cbind(
    as.data.frame(dt[, X_base, with = FALSE]),
    as.data.frame(ws_mm),
    as.data.frame(country_mm)
  ))
  
  # Add source E.coli if needed (for diarrhea model)
  if (include_source_ecoli) {
    X <- cbind(X, log_WQ27 = dt$log_WQ27)
  }
  
  return(X)
}

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

get_summary <- function(dt) {
  cat("\n=== DATA SUMMARY ===\n\n")
  
  cat("Observations:", nrow(dt), "\n\n")
  
  cat("OUTCOMES:\n")
  cat("  Diarrhea:", sum(!is.na(dt$diarrhea)), "obs,", 
      sprintf("%.1f%%", mean(dt$diarrhea, na.rm = TRUE) * 100), "positive\n")
  cat("  log(1+WQ26): Mean =", sprintf("%.2f", mean(dt$log_WQ26, na.rm = TRUE)),
      ", SD =", sprintf("%.2f", sd(dt$log_WQ26, na.rm = TRUE)), "\n\n")
  
  cat("TREATMENTS:\n")
  cat("  Any treatment:", sprintf("%.1f%%", mean(dt$any_treatment, na.rm = TRUE) * 100), "\n")
  cat("  Boil:", sprintf("%.1f%%", mean(dt$boil, na.rm = TRUE) * 100), "\n")
  cat("  Chlorine:", sprintf("%.1f%%", mean(dt$chlorine, na.rm = TRUE) * 100), "\n")
  cat("  Filter:", sprintf("%.1f%%", mean(dt$filter, na.rm = TRUE) * 100), "\n\n")
  cat("  Other:", sprintf("%.1f%%", mean(dt$other_treat, na.rm = TRUE) * 100), "\n\n")
    
  cat("CONFOUNDER:\n")
  cat("  log(1+WQ27): Mean =", sprintf("%.2f", mean(dt$log_WQ27, na.rm = TRUE)),
      ", SD =", sprintf("%.2f", sd(dt$log_WQ27, na.rm = TRUE)), "\n")
}