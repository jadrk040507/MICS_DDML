# MICS Python analyses

This guide is for researchers who want to run the analysis or understand its main choices. You do not need to read the implementation files to run the published workflow.

## Start here

The project environment is `.venv` at the repository root. Open a terminal in `Do file/Python` and run one of these commands:

```bash
uv run python 01_run_analysis.py
```

This runs the primary **clustered** analysis: ATE, ATT, sensitivity analysis, and GATE. The run can take a long time because it fits many models and also repeats the main estimates for selected countries. It saves completed model fits as checkpoints so an interrupted run can reuse valid work.

Other choices:

```bash
uv run python 01_run_analysis.py unclustered  # robustness analysis with ordinary folds
uv run python 01_run_analysis.py all          # clustered analysis, then unclustered robustness
```

Run from `Do file/Python`; `uv` finds the project environment in the repository root. The input data are read from `Data/3. Final/`. Analysis outputs are written under `Output/`.

## What clustered and unclustered mean

- **Clustered (C)** keeps observations from the same primary sampling unit (PSU) together when assigning cross-fitting folds and uses PSU-clustered standard errors.
- **Unclustered (U)** assigns folds at the observation level and uses ordinary standard errors.
- C is the primary specification. U is a robustness specification.

## Run one stage

The numbered scripts let you run a stage without launching the full sequence. Run them from this directory, in order when starting from scratch:

| Script | What it does | Output folder |
|---|---|---|
| `02_ate_c.py` | Clustered average treatment effect (ATE), main tables, and selected-country estimates | `Output/ATE_C/` |
| `03_att_c.py` | Clustered average treatment effect on the treated (ATT), main tables, and selected-country estimates | `Output/ATT_C/` |
| `04_sensitivity_c.py` | Clustered sensitivity benchmarks for ATE and ATT | `Output/ATE_C/`, `Output/ATT_C/` |
| `05_gate_c.py` | Clustered group average treatment effects (GATE) for ATE and ATT | `Output/ATE_C/`, `Output/ATT_C/` |
| `06_ate_u.py` | Unclustered ATE and selected-country estimates | `Output/ATE_U/` |
| `07_att_u.py` | Unclustered ATT and selected-country estimates | `Output/ATT_U/` |
| `08_sensitivity_u.py` | Unclustered sensitivity benchmarks for ATE and ATT | `Output/ATE_U/`, `Output/ATT_U/` |
| `09_gate_u.py` | Unclustered GATE for ATE and ATT | `Output/ATE_U/`, `Output/ATT_U/` |

The sensitivity and GATE scripts first load or fit the baseline models they need, then calculate their stage's results. If compatible model checkpoints already exist, they can be reused. Running a numbered script does not require running `01_run_analysis.py` as well.

## Where to look in the code

The numbered scripts are small entry points. Most researchers only need the README and those scripts. The analysis implementations are long because they keep each estimand's full workflow and its documentation together; use the section map near the top of `_ate_impl.py` or `_att_impl.py` to jump to a specific task.

| File | Purpose |
|---|---|
| `_analysis_runner.py` | Chooses and runs stages in order |
| `_ate_impl.py` | ATE data preparation, estimation, inference, sensitivity, and GATE workflow |
| `_att_impl.py` | ATT workflow, including the treated target population |
| `_analysis_data.py` | Shared complete-case samples, model-ready frames, and selected-column data loading |
| `_cross_fitting.py` | Shared reproducible observation and PSU fold construction |
| `_ddml_engine.py` | Shared Super Learner, clustered inference, and GATE projection |
| `_checkpoint_io.py`, `_model_checkpoint_compat.py`, `_provenance.py` | Save, load, validate, and identify reusable model and sensitivity checkpoints |
| `_sensitivity_groups.py`, `_sensitivity_scale.py` | Define omitted control groups and convert sensitivity measures |
| `_joint_inference.py` | Shared joint-inference calculations |
| `_compare_ate_att_atu.py` | Auxiliary historical ATE/ATT/ATU comparison report; see its warning below |

## A few terms

- **ATE:** average effect of a treatment in the analysis population.
- **ATT:** average effect for the treated population. Here, the ATT target is households reporting any water treatment.
- **IRM:** interactive regression model, used here for the binary treatment comparison.
- **APOS:** average potential outcome by treatment level; contrasts compare a treatment level with no treatment.
- **Cross-fitting:** estimates are evaluated on folds that were held out while fitting the prediction models. With clustered folds, all observations from one PSU stay in the same fold.
- **Super Learner:** combines predictions from several candidate models using weights chosen from the data.
- **Checkpoint:** a saved fitted model or completed sensitivity calculation that can be reused on a later run if its inputs and analysis settings still match.
- **GATE:** group average treatment effect, showing how estimated effects vary across prespecified groups.
- **PSU:** primary sampling unit, the survey cluster used for fold assignment and clustered standard errors.

## How to read the sensitivity results

Sensitivity analysis asks how strong an omitted confounder would need to be to change the estimated result. It benchmarks prespecified groups of controls using the fitted baseline model; it does **not** permanently remove each group and re-estimate ATE or ATT.

The benchmark checks nine control groups in the child sample and eight in household samples. Examples include wealth, urban residence, water source, E. coli decile, household composition, toilet, and country fixed effects. Child analyses also include child age and sex. When a group is categorical, all encoded indicators for that group are omitted together. PSU identifiers are not control groups.

For each benchmark, the saved raw sensitivity inputs are `cf_y`, `cf_d`, and `rho`. The analysis also reports two equal-strength point-bias measures:

- `r_equiv_empirical` uses the absolute correlation `abs(rho)` returned by DoubleML.
- `r_equiv_adversarial` sets `abs(rho)=1`, the strongest-correlation benchmark.

Compare these measures with the point robustness value (RV) as a descriptive scale. RV-alpha also accounts for uncertainty in the bound, so comparing a benchmark only with RV-alpha does not establish whether a confidence interval includes zero.

`results_sensitivity_groups.csv` contains results for each outcome, estimand, and omitted group. `results_sensitivity_summary.csv` and `table_sensitivity_main.tex` report the median sensitivity strength across groups alongside baseline RV and RV-alpha. Pickle files retain detailed parameters.

Some finite-sample benchmarks can produce inputs outside the formula's valid range. In that case, the raw values are kept, a warning is printed, and only the affected derived measure is set to `NaN`; the run continues. The empirical and strongest-correlation measures are checked separately. At the boundary `cf_d=1`, the calculation uses its mathematical limit rather than shifting the value inward. Check warnings and raw benchmark columns when interpreting affected rows.

Each completed group/model/outcome/fold-mode benchmark has its own checkpoint in the relevant `checkpoints/` folder. Rerunning sensitivity reuses valid completed groups, recalculates missing or invalid ones, and rebuilds the tables. Its fingerprint includes the sensitivity calculation code, so a change to those calculations refreshes sensitivity checkpoints without forcing compatible baseline model fits to be repeated.

## Outputs and reproducibility

The C/U output folders contain result tables, LaTeX tables, manifests, and a `checkpoints/` subfolder. A model checkpoint is reused only when the relevant data, locked environment, Python version, model engine, shared data/fold helpers, worker settings, and estimation settings match. This code refactor changes the model fingerprint, so existing checkpoint files remain on disk but will not be loaded automatically by the refactored workflow. Sensitivity checkpoints use a separate fingerprint derived from the model fingerprint and sensitivity-specific code. The manifest records the fingerprints and checkpoint files used.

## Historical ATE/ATT/ATU comparison

`_compare_ate_att_atu.py` derives ATU from paired ATE and ATT checkpoints within each cross-fitting repetition. It accounts for the estimated treated share and its covariance with ATE and ATT. This report has not been run end to end with the full fitted models. Before relying on its joint table, check that its marginal ATE and ATT standard errors agree with the standalone estimates. The selected-country report also needs matching selected-country checkpoints. It can read legacy model checkpoints from `Output/ATE/` and `Output/ATT/` when available.
