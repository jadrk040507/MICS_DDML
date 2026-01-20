# MICS Double Machine Learning Analysis

Comprehensive Double Machine Learning (DoubleML) analysis for estimating the causal effect of water treatment on E.coli presence indicators using Multiple Indicator Cluster Surveys (MICS) data.

## 📋 Project Overview

This project implements a rigorous causal inference analysis using the DoubleML framework to estimate the treatment effect of water treatment (`water_treatment`) on two E.coli risk indicators:
- **SomeRiskHome**: Indicator for some E.coli risk
- **VeryHighRiskHome**: Indicator for very high E.coli risk

## 📁 Project Structure

```
MICS/
├── 01_data.ipynb                    # Data preprocessing
├── 02_doubleml_analysis.ipynb       # Main DoubleML analysis
├── 03_robustness_analysis.ipynb     # Robustness verification
├── mics_clean.csv                   # Cleaned dataset
├── README.md                        # This file
└── pyproject.toml                   # Python dependencies
```

## 📓 Notebook Descriptions

### 1. `01_data.ipynb` - Data Preprocessing
**Purpose:** Load and prepare raw MICS data for analysis

**Key Steps:**
- Load raw `mics.csv` data
- Define outcome variables, treatment variable, and covariates
- Handle categorical and ordinal variable encoding
- One-hot encode categorical variables (Region, country_cat, WS1_g)
- Map ordinal variables (helevel, windex5, RiskSource, water_carrier_edu)
- Export cleaned dataset to `mics_clean.csv`

**Important:** NaN values are preserved (not imputed) in this preprocessing step. They will be dropped in subsequent analyses as DoubleML requires complete cases.

---

### 2. `02_doubleml_analysis.ipynb` - Main DoubleML Analysis
**Purpose:** Comprehensive DoubleML causal inference analysis

**Analysis Structure:**

#### For Each Outcome (SomeRiskHome, VeryHighRiskHome):

**A. Base Models**
- **Covariates:** windex5, helevel, country_cat, urban, WS1_g, wq27_decile
- **Models:** PLR (Partial Linear Regression) and IRM (Interactive Regression Model)
- **Output:** Combined LaTeX table with both PLR and IRM results

**B. Extended Models**
- **Covariates:** All available variables (except outcomes and treatment)
- **Models:** PLR and IRM
- **Output:** Combined LaTeX table with both PLR and IRM results

**C. Subsample Analysis by RiskSource**
For each RiskSource category ("No risk", "Moderate to high risk", "Very high risk"):
- **4 Models per subsample:** Base PLR, Base IRM, Extended PLR, Extended IRM
- **Output:** Separate LaTeX table for each RiskSource category

**Technical Details:**
- **ML Algorithm:** XGBoost (tree-based, no scaling needed)
- **Hyperparameter Tuning:** Optuna with 20 trials (15 for subsample analysis)
- **Cross-validation:** Built into DoubleML framework
- **Total Models:** 20 models (10 per outcome variable)

---

### 3. `03_robustness_analysis.ipynb` - Robustness Verification
**Purpose:** Verify robustness of nuisance model specifications

**Analysis Structure:**

#### For Each Target Variable:

**A. Outcome Models (Y)**
- **Target 1:** SomeRiskHome
- **Target 2:** VeryHighRiskHome

**B. Treatment/Propensity Model (T)**
- **Target:** water_treatment

**Models Tested (10 total):**

*Models with Scaled Features:*
1. **OLS** (benchmark) - Linear Regression
2. **Logistic Regression** - L1/L2 regularization
3. **Lasso** - L1 regularization
4. **Ridge** - L2 regularization
5. **ElasticNet** - Combined L1/L2
6. **Naive Bayes** - Gaussian NB

*Models with Unscaled Features (Tree-based):*
7. **Random Forest** - Ensemble of decision trees
8. **XGBoost** - Gradient boosting
9. **LightGBM** - Light gradient boosting
10. **AdaBoost** - Adaptive boosting

**Hyperparameter Tuning:**
- **Method:** RandomizedSearchCV (faster than GridSearch)
- **Cross-validation:** 3-fold CV
- **Iterations:** 10 per model

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (Area Under ROC Curve)

**Total Models:** 30 models (10 models × 3 targets)

---

## 🔑 Key Variables

### Treatment Variable
- `water_treatment`: Binary indicator (0/1)

### Outcome Variables
- `SomeRiskHome`: Binary E.coli risk indicator
- `VeryHighRiskHome`: Binary high E.coli risk indicator

### Base Model Covariates
1. `windex5`: Wealth index (ordinal: Poorest to Richest)
2. `helevel`: Education level (ordinal: No education, Primary, Secondary+)
3. `country_cat`: Country categories (categorical)
4. `urban`: Urban/rural (binary)
5. `WS1_g`: Water source groups (numeric)
6. `wq27_decile`: Water quality decile (ordinal)

### Extended Model Covariates
All available covariates including:
- Binary indicators: household composition, water services, sanitation, hygiene
- Ordinal variables: education, wealth, water carrier education
- Categorical variables: region, country, water source groups
- Seasonal indicators: rainy season

### Subsample Variable
- `RiskSource`: Categories for subsample analysis
  - "No risk"
  - "Moderate to high risk"
  - "Very high risk"

---

## 🚀 Usage Instructions

### 1. Prerequisites
```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm doubleml optuna
```

Or use the project's pyproject.toml:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Run Analysis

**Step 1: Data Preprocessing**
```bash
jupyter notebook 01_data.ipynb
```
- Run all cells to create `mics_clean.csv`

**Step 2: DoubleML Analysis**
```bash
jupyter notebook 02_doubleml_analysis.ipynb
```
- Run all cells to execute complete DoubleML analysis
- Expected runtime: 30-60 minutes (depending on hardware)
- Outputs: LaTeX tables for all model specifications

**Step 3: Robustness Analysis**
```bash
jupyter notebook 03_robustness_analysis.ipynb
```
- Run all cells to test 10 different ML algorithms
- Expected runtime: 20-40 minutes
- Outputs: Performance comparison tables and LaTeX tables

---

## 📊 Expected Outputs

### From `02_doubleml_analysis.ipynb`:
1. **Base Models Tables** (2): SomeRiskHome and VeryHighRiskHome
2. **Extended Models Tables** (2): SomeRiskHome and VeryHighRiskHome
3. **Subsample Tables** (6): 3 RiskSource categories × 2 outcomes
4. **Total:** 10 LaTeX tables with treatment effect estimates

### From `03_robustness_analysis.ipynb`:
1. **Outcome Model Tables** (2): SomeRiskHome and VeryHighRiskHome
2. **Propensity Model Table** (1): water_treatment
3. **Cross-Model Comparison** (1): All models across all targets
4. **Total:** 4 LaTeX tables with performance metrics

---

## 📈 Interpretation Guide

### DoubleML Results
Each table includes:
- **Coefficient**: Treatment effect estimate (ATE - Average Treatment Effect)
- **Std. Error**: Standard error of the estimate
- **95% CI**: Confidence interval [lower, upper]
- **p-value**: Statistical significance

**Interpretation:**
- Positive coefficient → water treatment increases E.coli risk
- Negative coefficient → water treatment decreases E.coli risk
- p-value < 0.05 → statistically significant effect

### Robustness Results
Each table shows model performance:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1**: Harmonic mean of precision and recall
- **AUC**: Discriminative ability (higher is better)

**Good performance indicators:**
- F1 > 0.70
- AUC > 0.70
- Consistent performance across models

---

## 🛠️ Technical Notes

### Why No Scaling for DoubleML?
- XGBoost (tree-based) doesn't require feature scaling
- Preserves interpretability of original variable scales
- Standard practice for gradient boosting methods

### Why Scaling for Robustness?
- Linear models (OLS, Logistic, Lasso, Ridge, ElasticNet) benefit from scaling
- Naive Bayes assumes features are on similar scales
- Tree-based models (RF, XGBoost, LightGBM, AdaBoost) don't need scaling

### NaN Handling
- Preprocessing: NaNs preserved (no imputation)
- Analysis: Complete cases used (`.dropna()`)
- Rationale: DoubleML requires complete data; alternative NaN handling can introduce bias

### Hyperparameter Tuning
- **DoubleML:** Optuna (Bayesian optimization, 20 trials)
- **Robustness:** RandomizedSearchCV (10 iterations, 3-fold CV)
- Trade-off: Speed vs. thoroughness (Optuna more thorough but slower)

---

## 📚 References

### DoubleML Framework
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML - An Object-Oriented Implementation of Double Machine Learning in Python. *Journal of Machine Learning Research*, 23(53), 1-6.

### Causal Inference
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

### Machine Learning
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NIPS*, 3146-3154.

---

## 🤝 Contributing

This is a research project. For questions or suggestions, please contact the project maintainer.

---

## 📄 License

[Specify your license here]

---

## ✅ Checklist for Complete Analysis

- [x] Data preprocessing complete (`01_data.ipynb`)
- [ ] DoubleML base models estimated
- [ ] DoubleML extended models estimated
- [ ] Subsample analysis by RiskSource completed
- [ ] Robustness analysis with 10 models completed
- [ ] All LaTeX tables generated
- [ ] Results interpreted and documented
- [ ] Findings ready for publication/presentation

---

**Last Updated:** January 2026  
**Project Status:** Implementation Complete - Ready for Analysis
