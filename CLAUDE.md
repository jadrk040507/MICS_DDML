# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Causal inference pipeline estimating the effect of household water treatment on E. coli contamination and child diarrhea across MICS (Multiple Indicator Cluster Surveys) countries. Uses **Double/Debiased Machine Learning (DDML)** via the `doubleml` library with seven ML learners.

Two datasets:
- **HH** (`MASTER_MICS_FINAL.dta`): household-level, E. coli outcomes (`SomeRiskHome`, `VeryHighRiskHome`)
- **U5** (`MASTER_MICS_FINAL_U5.dta`): child-level, diarrhea outcome

All Python analysis code lives in `Do file/python/` (alongside the Stata `.do` scripts under `Do file/`). Run the `run_*.py` scripts from that directory, or ensure it is on `sys.path`.

## Commands

### Environment Setup
```powershell
# Activate virtualenv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r "Do file/python/requirements.txt"
```

### Run Analysis
```powershell
# Main analysis (all 7 learners, both datasets)
cd "Do file/python"
python run_main.py

# Subset of learners (faster)
python run_main.py --learners ols lasso rf

# Override CV settings
python run_main.py --n-folds 3 --n-rep 1

# Parallel learners
python run_main.py --parallel 4

# Robustness checks
python run_robustness.py

# Subgroup analysis (by RiskSource)
python run_subgroups.py

# Falsification test (RiskSource=0 placebo)
python run_falsification.py

# CATE by education subgroup
python run_cate_education.py

# Multi-treatment APOS (boil/chlorine/filter/other vs none, full sample)
python run_apos.py
```

### Tests
```powershell
# Run all tests (from repo root; conftest.py puts Do file/python on sys.path)
pytest "Do file/python/tests/"

# Single test file
pytest "Do file/python/tests/test_models.py"

# Single test
pytest "Do file/python/tests/test_models.py::test_estimate_effect_ols"
```

## Architecture

### Module Roles

| File | Role |
|------|------|
| `config.py` | Single source of truth for all settings: paths, outcomes, treatments, confounder groups, learner names, CV parameters |
| `data.py` | Loads `.dta` files via `pyreadstat`; constructs all derived variables; `create_model_matrix()` expands confounder groups to column arrays |
| `learners.py` | Defines 7 ML learners (OLS, Lasso, Ridge, ENet, RF, XGBoost, Stacked). `create_learners()` for continuous/LPM nuisance; `create_learners_for_binary()` wraps classifiers in `ProbaRegressor` to return bounded probabilities |
| `models.py` | Core estimation: `estimate_effect()` runs `DoubleMLIRM` with checkpoint caching; `run_analysis()` loops over outcome×treatment×learner combinations, reusing model matrices per pair |
| `apos.py` | Multi-treatment `DoubleMLAPOS` (full-sample AIPW): `estimate_apos()`/`run_apos()` estimate E[Y(d)] for every WQ15_g level (stacked nuisances, multi-class propensity) and return ATE(d vs none) via `causal_contrast` (i.i.d.; APOS has no clustered SE in doubleml 0.11.x) |
| `runners.py` | Shared boilerplate (`setup_environment`, `load_data`, `select_learners`, `save_results`) imported by all `run_*.py` scripts |
| `robustness.py` | Falsification, coefficient stability (progressive confounder addition), leave-one-out confounder analysis |
| `tables.py` / `figures.py` / `diagnostics.py` | Output generation: LaTeX tables, matplotlib figures, diagnostic plots |

### Estimation Flow

1. `prepare_hh_data()` / `prepare_u5_data()` → constructs all model variables
2. `create_model_matrix()` → expands `BASE_CONFOUNDERS` dict into numpy array `X`
3. `prepare_model_data()` → filters complete cases, checks treatment variation
4. `estimate_effect()` → fits `DoubleMLIRM` with 3-attempt fallback (default folds → 2 folds → no clustering); saves/loads `.pkl` checkpoints in `Output/checkpoints/`
5. Results aggregated into `Output/results_<tag>.csv` + `.pkl`

### Key Design Decisions

- **Checkpoint system**: each `(dataset, outcome, treatment, learner, confounder_set)` combination serialized to `Output/checkpoints/<name>.pkl`. Re-runs skip completed specs automatically.
- **ProbaRegressor**: classifier wrapper in `learners.py` that makes `predict()` return `predict_proba()[:, 1]`, keeping binary outcome nuisance functions bounded in `[0,1]` for DoubleML's `ml_g`.
- **Confounder groups**: `config.py` defines named groups (e.g., `"wealth"`, `"country"`). `create_model_matrix()` knows that `"water_source"` → expand `ws1g_*` columns (drop first) and `"country"` → expand `country_*` dummies. Add new confounder groups in `config.py` only.
- **Clustered SEs**: `CLUSTER_VAR = "Cluster_var"` (PSU). DoubleML receives `cluster_vars` array; fallback drops clustering if fold splits produce degenerate clusters.
- **`single_method` restriction**: specific treatments (e.g., `treat_boil`) are estimated on households using exactly one treatment method, controlled via `restrict_single_method=True` in `run_analysis()`.

### Confounder Group Structure

`BASE_CONFOUNDERS` in `config.py` is a dict of `{group_name: [col_names]}`. For HH analysis, this is used as-is. For U5, `U5_ADDITIONAL_CONFOUNDERS` (`child_age`, `child_sex_male`) is merged in. `ROBUSTNESS_CONFOUNDERS` adds water storage and handwashing variables.

### Tests Use Fake Data

Tests in `Do file/python/tests/` construct synthetic data via `make_fake_data()` rather than loading the actual `.dta` files. All data-construction helpers (`_construct_common_variables`, etc.) are tested in isolation. Run with `skip_checkpoint=True` to avoid disk I/O during tests.
