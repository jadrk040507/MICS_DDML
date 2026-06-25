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

# Scripts are numbered by workflow order: IRM -> APOS -> CATE -> GATE,
# followed by supporting analyses (50 predictors, 60 sensitivity).

# 1. IRM (ANY treatment only, water_treatment vs none, full sample, all 7
#    learners) + sensitivity (RV/RVa). Specific-method effects are reported by
#    APOS, not IRM. results_main.pkl is consumed by step 2.
python 10_run_irm.py
python 10_run_irm.py --learners ols lasso rf       # subset (faster)
python 10_run_irm.py --n-folds 3 --n-rep 1         # override CV
python 10_run_irm.py --parallel 4                  # parallel learners

# 2. Multi-treatment APOS (boil/chlorine/filter/other vs none, full sample).
#    Loads the any-treatment IRM from step 1 and writes the two headline results
#    tables: table_results_ecoli.tex and table_results_diarrhea.tex, each with
#    the IRM any-treatment row on top of the APOS contrast panel.
python 20_run_apos.py

# 3. CATE — projection-based conditional effects (smooth B-spline curves over
#    wscore (continuous wealth, df=5) and num_children (raw count 0-10, df=3));
#    writes one PNG per spec to the ROOT Figures/ directory
python 30_run_cate.py

# 4. GATE — projection-based group effects (RiskSource, education, wealth
#    quintile, country, sanitation, water source, binned child count, +child
#    age for U5) with joint (uniform) confidence bands
python 40_run_gate.py

# Supporting analyses
python 50_run_treatment_predictors.py              # which covariates predict adoption
python 60_run_sensitivity.py                       # Chernozhukov RV + confounder benchmarking
python 60_run_sensitivity.py --learner stacked     # headline-exact RV (slow; default lasso)
```

> CATE/GATE in `30`/`40` are the **projection-based** estimands (`model.cate()` /
> `model.gate()` on a single fitted IRM). Every discrete-cell moderator is a GATE
> group; CATE is reserved for the two genuinely continuous/well-supported
> moderators (wscore, num_children).

### Tests
No automated test suite currently ships in the package — the former
`Do file/python/tests/` was removed in the consolidation (commit `1c9f2e5`).
Verify edits by importing the package instead:
```powershell
cd "Do file/python"
python -c "import _config,_data,_learners,_models,_tables,_figures,_apos,_heterogeneity_apos,_runners; print('OK')"
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
| `_heterogeneity_apos.py` | Projection-based GATE/CATE on the APOS estimand (Semenova–Chernozhukov BLP). `fit_base_apos()` fits ONE base `DoubleMLAPOS` (clustered per dataset — U5 on HHID); `estimate_gate_apos()` projects onto group dummies; `estimate_cate_apos_spline()` projects onto a B-spline basis; `estimate_gapo_levels()` / `estimate_capo_spline()` give the per-level GAPO/CAPO variants. Joint (uniform) + pointwise bands. (The earlier IRM-based `heterogeneity.py` was removed in the consolidation; CATE/GATE now run entirely through this APOS path.) |
| `60_run_sensitivity.py` | Chernozhukov et al. (2022) sensitivity for every spec (3 outcomes × any + specific treatments). Refits a **named** `DoubleMLIRM` (so confounder groups are addressable), reports RV / RV_alpha, and runs `model.sensitivity_benchmark()` per confounder group → cf_y / cf_d / delta_theta ("is RV larger than the strongest observed confounder?"). Default learner `lasso` (fast; cf are design-stable); `--learner stacked` for headline-exact RV. Writes `results_sensitivity.csv` + `table_sensitivity_benchmark.tex` |
| `runners.py` | Shared boilerplate (`setup_environment`, `load_data`, `select_learners`, `save_results`) imported by all `run_*.py` scripts |
| `tables.py` / `figures.py` | Output generation. `create_results_table` builds the two headline landscape tables (outcomes × 7 learners, coef over (se)): the **any-treatment IRM is the top row**, the APOS method-vs-none contrasts (Boil/Chlorine/Filter/Other) are the rows below it. Called twice from `20_run_apos.py` — `table_results_ecoli.tex` (SomeRiskHome + VeryHighRiskHome) and `table_results_diarrhea.tex` (diarrhea, standalone). Stacked base-learner meta-weights (g \| m) shown in the stacked columns; RV / RV_α inline per spec. `create_sensitivity_benchmark_table` is the same landscape layout for the confounder benchmark (per outcome: cf_y / cf_d / Δθ for each group; rows = treatments; confounder order **derived from `BASE_CONFOUNDERS`** so it can't drift). `figures.plot_cate_curves` writes smooth CATE(x) PNGs to the ROOT `Figures/`. All landscape tables need `\usepackage{booktabs,pdflscape,graphicx}`. |

### Estimation Flow

1. `prepare_hh_data()` / `prepare_u5_data()` → constructs all model variables
2. `create_model_matrix()` → expands `BASE_CONFOUNDERS` dict into numpy array `X`
3. `prepare_model_data()` → filters complete cases, checks treatment variation
4. `estimate_effect()` → fits `DoubleMLIRM` at exactly the requested `(n_folds, n_rep, cluster_vars)` — a **single attempt, no silent downgrade**: if the fit fails it returns `None` (spec reported failed) rather than re-fitting with fewer folds or clustering dropped; saves/loads `.pkl` checkpoints in `Output/checkpoints/`
5. Results aggregated into `Output/results_<tag>.csv` + `.pkl`

### Key Design Decisions

- **Checkpoint system**: each IRM `(dataset, outcome, treatment, learner, confounder_set)` combination serialized to `Output/Checkpoints/<name>.pkl`. APOS caches its result list per `(dataset, outcome, learner)` as `apos_<dataset>_<outcome>_<learner>.pkl`. Re-runs skip completed specs automatically; delete the files to force a refit (e.g. after changing confounders, folds, or learner definitions).
- **ProbaRegressor**: classifier wrapper in `learners.py` that makes `predict()` return `predict_proba()[:, 1]`, keeping binary outcome nuisance functions bounded in `[0,1]` for DoubleML's `ml_g`.
- **Stacking weights**: for the `stacked` learner, IRM/APOS fit with `store_models=True`; `_extract_stacking_weights()` averages each base learner's (lasso/ridge/rf) meta-weight across cross-fit folds/reps, separately for the outcome model `g` and propensity `m`, and stores `weights_g`/`weights_m` on the result for the tables.
- **Confounder vector (mirrors the Stata spec exactly)**: `BASE_CONFOUNDERS` = `i.windex5 i.helevel i.country_cat i.urban i.WS1_g Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile`; `U5_ADDITIONAL_CONFOUNDERS` adds `i.age i.male` (child age/sex) for U5 ONLY. `create_model_matrix()` expands the `i.` categoricals to drop-first dummies: `"water_source"`→`ws1g_*`, `"country_cat"`→`country_cat_*`, `"wq27_decile"`→`wq27_d_*`, `"child_age"`→`child_age_*`. The SAME vector is used by IRM, APOS, CATE, GATE, and the sensitivity benchmark (`60_run_sensitivity.py`). `RiskSource` (source E.coli) and `num_children` are NOT controls: `RiskSource` is a GATE splitter and `num_children` is a CATE moderator (raw count → spline) / GATE group (`num_children_bin`); child counts are captured for adjustment by the binary demographic indicators.
- **Clustered SEs (per dataset)**: `CLUSTER_VARS = {"HH": None, "U5": "HHID"}` via `cluster_var_for()`. The HH (E.coli) analysis is i.i.d. (one row per household, no HHID); the U5 (child) analysis clusters on household (HHID). This applies to IRM, APOS (where supported), **and** the GATE/CATE heterogeneity layer. `estimate_effect()` does **not** downgrade folds or drop clustering on failure — a failed fit is reported as `None`.
- **IRM = any treatment only; specific methods = APOS**: `10_run_irm.py` estimates ONLY the any-treatment ATE (`water_treatment` vs none, full sample) for every outcome, plus its sensitivity. Specific-method effects (boil/chlorine/filter/other vs none) are reported exclusively by APOS (`20_run_apos.py`), the efficient full-sample AIPW estimator. The any-treatment IRM is loaded from `results_main.pkl` by step 2 and rendered as the **top row** of the combined results tables, above the APOS contrast rows. (The `vs_none` / `restrict_single_method` specific-method IRM paths in `prepare_model_data` are retained but no longer invoked.)
- **Results split by outcome (three reporting sections)**: (1) **water-treatment adoption** → `table_treatment_predictors.tex` (`50_run_treatment_predictors.py`); (2) **E.coli contamination** → `table_results_ecoli.tex` (SomeRiskHome + VeryHighRiskHome); (3) **diarrhea** → `table_results_diarrhea.tex` (standalone) plus the sensitivity analyses (`60_run_sensitivity.py`). Tables 2 and 3 share the IRM-on-top-of-APOS layout from `create_results_table`.
- **CATE vs GATE moderator split** (`config.py`): `CATE_MODERATORS` holds only genuinely continuous / well-supported moderators projected onto a B-spline (`wscore` df=5, `num_children` raw 0-10 df=4 — patsy `bs` needs df≥degree+1 with `include_intercept`). Every discrete-cell moderator (education, child age, sanitation, water source, binned child count) is a `GATE_GROUPS` entry instead (one-hot via `model.gate()`).
- **Sensitivity benchmarking** (`60_run_sensitivity.py`): RV / RV_alpha come "free" from a fit; the per-group cf_y/cf_d/Δθ require named covariates, so this script refits its own `DoubleMLData` (DataFrame backend with real column names) rather than reusing the `from_arrays` checkpoints. Headline finding: the dominant confounder is **country** (cf_d≈0.67), then **source E.coli** (cf_y≈0.20) — NOT wealth; the E.coli any-treatment base (CATE/GATE base) is robust to a confounder as strong as country, diarrhea is not (null ATE → RV≈0).

### Confounder Group Structure

`BASE_CONFOUNDERS` in `config.py` is a dict of `{group_name: [col_names]}`. For HH analysis, this is used as-is. For U5, `U5_ADDITIONAL_CONFOUNDERS` (`child_age`, `child_sex_male`) is merged in.

