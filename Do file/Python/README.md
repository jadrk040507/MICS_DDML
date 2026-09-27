# MICS DoubleML analysis

This directory contains the Python workflow for the MICS water treatment analysis. Run commands here; `uv` discovers the locked environment at the repository root.

## Run the analysis

```bash
uv run python run_analysis.py
```

The default runs clustered cross-fitting for ATE and ATT and produces effects, sensitivity results, and GATE results. A completed current-fingerprint checkpoint is reused automatically.

Use one command to select another scope:

```bash
uv run python run_analysis.py unclustered
uv run python run_analysis.py all
uv run python run_analysis.py --estimand ate
uv run python run_analysis.py --estimand att --stage effects
uv run python run_analysis.py clustered --stage sensitivity
uv run python run_analysis.py clustered --stage gate
```

Arguments:

- `fold_mode`: `clustered` (default), `unclustered`, or `all`.
- `--estimand`: `ate`, `att`, or `both` (default).
- `--stage`: `effects`, `sensitivity`, `gate`, or `all` (default).

Clustered folds keep every primary sampling unit (PSU) in one fold and report PSU clustered inference. Unclustered folds split observations and report ordinary inference. The clustered specification is primary; unclustered results are a robustness check.

## Five-file map

| File | Purpose |
|---|---|
| `run_analysis.py` | Command line choices and stable execution order |
| `analysis.py` | Prespecified controls, learners, ATE/ATT strategies, model fitting, and workflow order |
| `ddml.py` | Data frames, cross-fitting, Super Learner, clustered inference, sensitivity framework, and common GATE calculations |
| `reporting.py` | Publication-stage boundaries, sensitivity groups, and sensitivity-scale calculations |
| `artifacts.py` | Current checkpoint paths, atomic writes, lazy model bundles, and provenance fingerprints |

ATE and ATT use the same orchestration. Their differences are explicit in `AnalysisSpec`: IRM score (`ATE` or `ATTE`), ATT target weights for multivalued APOS, and the ATT weighted-score GATE strategy.

## Data and outputs

Inputs are read from `Data/3. Final/`. Outputs are written to:

- `Output/ATE/` for ATE results;
- `Output/ATT/` for ATT results.

Fold mode is recorded in checkpoint and result specifications inside those canonical estimand directories.

Each output directory contains result files, LaTeX tables, a manifest, and `checkpoints/`. Existing output and checkpoint files are not deleted by the workflow.

## Reproducibility and checkpoints

Checkpoint filenames include a fingerprint of the input data, locked environment, Python version, analysis settings, worker settings, and the four implementation modules used by estimation. Sensitivity checkpoints have a separate fingerprint derived from the model fingerprint and sensitivity code.

This consolidation changes the fingerprint once. Earlier checkpoint files remain on disk but are not searched or loaded. The workflow reads only the exact current-fingerprint path. A corrupt current checkpoint raises a load error so the problem is visible.

Quick sample and full-run checkpoints have different suffixes. Random seeds, fold counts, treatment levels, categorical encoding order, PSU isolation, propensity clipping, and learner definitions are recorded in the analysis and manifest.

## Statistical terms

- **ATE:** average treatment effect in the analysis population.
- **ATT:** average treatment effect among households reporting any water treatment.
- **IRM:** interactive regression model for the binary any-treatment comparison.
- **APOS:** average potential outcomes for treatment levels; reported contrasts compare levels 1–3 with level 0.
- **GATE:** group average treatment effect for prespecified source-water groups.
- **Super Learner:** convex combination of the prespecified prediction models.

Sensitivity analysis benchmarks omitted groups of controls. Household analyses use eight groups; child analyses add age and sex as a ninth group. All encoded indicators for a categorical control are omitted together, and PSU identifiers never enter the benchmark controls.

## Validation

The unit suite is fast and does not launch full estimation:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```
