# Consolidated Python Workflow Design

## Purpose

Make the MICS DDML Python workflow easy to run and audit without requiring a
researcher to navigate duplicate ATE and ATT implementations, empty section
headings, legacy compatibility layers, or many one-purpose entry scripts.

The primary audience is an applied economist who needs to understand the
analysis choices and execution order without reading software infrastructure
code. The refactor must preserve the statistical estimands and published
workflow behavior.

## Success Criteria

- One documented command-line runner controls estimand, fold mode, and stage.
- ATE and ATT share one estimation workflow; only estimand-specific formulas
  remain separate.
- The active Python implementation consists of five clearly named modules.
- Empty sections, compatibility aliases, legacy checkpoint lookup, obsolete
  wrappers, and historical Python copies are removed.
- Complete-case samples, controls, treatment levels, fold assignments, ATT
  weights, scores, clustered inference, sensitivity calculations, GATE rows,
  output names, and stage order retain their current behavior.
- A clean checkout passes the full unit suite without relying on untracked
  files.

## Active Module Structure

### `run_analysis.py`

The only executable entry point. It parses:

- fold mode: `clustered`, `unclustered`, or `all`;
- estimand: `ate`, `att`, or `both`;
- stage: `effects`, `sensitivity`, `gate`, or `all`.

Running without arguments preserves the current publication default:
clustered folds, both estimands, and all stages in documented order.

### `analysis.py`

Contains the economist-facing workflow and explicit analysis configuration:

- dataset and outcome specifications;
- household and child controls;
- ATE and ATT estimand configuration;
- data loading and complete-case construction;
- learner library definitions;
- IRM and APOS fitting orchestration;
- selected-country analysis;
- stage orchestration and manifest inputs.

An immutable `AnalysisSpec` or equivalent named configuration carries the
estimand-specific choices. ATE and ATT use the same workflow functions.
ATT-only functions remain explicit for treated-population weights and the
weighted orthogonal-score GATE calculation.

### `ddml.py`

Contains reusable statistical machinery:

- Super Learner convex weights and estimator classes;
- observation-level and PSU-grouped cross-fitting;
- clustered score aggregation and inference;
- clustered sensitivity framework construction;
- common GATE projection;
- coefficient formatting.

This module does not know output paths or publication table layouts.

### `reporting.py`

Contains output construction:

- main effect tables;
- Super Learner weight tables;
- sensitivity summaries and detailed benchmarks;
- GATE labels and tables;
- manifest assembly.

Functions receive explicit estimand labels and analysis results. Shared tables
must not branch on module identity or import separate ATE/ATT implementations.

### `artifacts.py`

Contains current artifact handling:

- provenance fingerprints;
- atomic joblib writes;
- checkpoint paths and validation;
- lazy loading of current outcome bundles.

Legacy, unversioned checkpoint lookup is removed. Checkpoints are reusable only
when their current fingerprint matches. The source restructuring intentionally
changes that fingerprint, so the first run after this refactor requires fresh
model checkpoints while leaving old files untouched on disk.

## Execution Flow

For each requested estimand and outcome, the runner calls one shared workflow:

1. Load only required columns and apply any country restriction.
2. Apply the existing complete-case rule and categorical encoding.
3. Construct deterministic observation or PSU-grouped folds.
4. Fit or load current-fingerprint IRM and APOS models.
5. Build requested effects, sensitivity, or GATE results.
6. Write tables, checkpoints, and the manifest under the current estimand
   directories, `Output/ATE/` and `Output/ATT/`; fold mode remains an explicit
   result/checkpoint specification rather than a different implementation.

The CLI expands `all` options into the same deterministic execution order used
by the current runner. NumPy random state is restored after each stage.

## Removed Files and Behaviors

The implementation removes:

- numbered runners `01_run_analysis.py` through `09_gate_u.py`;
- `_ate_impl.py` and `_att_impl.py`;
- small helpers absorbed into the five active modules;
- `_model_checkpoint_compat.py` and all legacy model-path fallback logic;
- `sys.modules` compatibility aliases and old public-name reexports;
- `_compare_ate_att_atu.py` and `_joint_inference.py`;
- the Python `archive/` directory, because Git history already preserves it;
- tests that exist only for removed compatibility or historical reports.

Old generated outputs and checkpoint files are not deleted by this change.
Raw and final input data remain immutable.

## Contracts That Must Not Change

- Household controls: wealth, urban status, water source, E. coli decile,
  under-five presence, child composition, toilet, and country fixed effects.
- Child samples add age and sex.
- Required control, outcome, treatment, country, and clustered-PSU fields retain
  the existing missing-value rule; optional metadata must not drop rows.
- Dummy reference categories and encoded-column order remain stable.
- Treatment levels remain `(0, 1, 2, 3, 98)` and reported levels remain
  `(0, 1, 2, 3)`.
- Clustered folds keep each PSU wholly in train or test; repeated folds use the
  same seed progression.
- ATT target population, propensity clipping, weights, score signs, and score
  dimensions remain unchanged.
- Checkpoint payload shapes and publication output filenames remain current,
  except that legacy checkpoint discovery is removed.
- The default runner remains clustered, both estimands, all stages.

## Error Handling

- Invalid CLI combinations fail with argparse help rather than silently
  selecting another mode.
- Invalid folds fail before model fitting when coverage, treatment support, or
  PSU isolation is violated.
- Missing or mismatched current checkpoints trigger estimation; corrupt current
  checkpoints raise a clear error.
- Atomic writes prevent partially serialized sensitivity checkpoints.
- Unsupported stages or estimands fail explicitly.

## Verification Strategy

Tests will cover:

- CLI defaults and option expansion;
- household and child control/sample construction;
- deterministic ordinary folds and PSU isolation;
- ATE and ATT IRM/APOS dispatch;
- ATT weights and weighted GATE scores;
- clustered coefficients, standard errors, intervals, and sample counts;
- sensitivity formulas and checkpoint validation;
- current checkpoint bundle shapes and provenance inputs;
- output filenames and stage order;
- absence of imports of removed modules;
- presence of only the documented runner and active modules.

The final gate is the complete unit suite from a detached clean worktree using
the locked project environment. The refactor will not run the expensive full
ATE/ATT estimation and will not modify data, existing checkpoints, or generated
research outputs.

## Reproducibility Consequence

The statistical specification remains unchanged, but code fingerprints change
because module paths and contents change. Existing checkpoints remain on disk
as historical artifacts and are not automatically reused. This is deliberate:
removing compatibility code makes checkpoint provenance simpler and prevents a
new workflow from silently loading objects created by an older implementation.
