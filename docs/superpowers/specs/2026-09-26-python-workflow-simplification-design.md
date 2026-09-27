# Python workflow simplification design

## Goal

Make the Python estimation code easier to navigate and maintain by removing exact ATE/ATT code duplication where a small, explicit shared interface is natural. Preserve the research specification, result formats, and separate ATE/ATT target populations. Keep the workflow readable to an applied researcher without requiring them to understand Python architecture.

## Current evidence

`_ate_impl.py` and `_att_impl.py` are about 3,600 and 3,950 lines. An AST comparison finds 20 top-level functions/classes with identical definitions, totaling about 1,429 repeated lines in each copy. Some are simple shared data and fold helpers; others are large orchestration functions that rely on estimand-module globals. Treating every identical function as a candidate for extraction would create a generic framework with hidden dependencies, so this change will focus on coherent low-level boundaries.

## Design

Keep `_ate_impl.py` and `_att_impl.py` as the two economist-facing estimand workflows. They remain responsible for ATE versus ATT choices, treatment target definitions, learner configuration, ATT weighting, and the order in which estimation and reporting run.

Move shared mechanics behind small functions with explicit inputs:

- Add `_analysis_data.py` for shared control selection, complete-case filtering, dummy encoding, and data loading. Pass the control lists and sample settings explicitly; do not have helper functions reach into ATE/ATT module globals.
- Add `_cross_fitting.py` for split validation, ordinary and PSU-clustered fold creation, and PSU fold metadata. Pass fold count, repetition count, seed, treatment values, and cluster IDs explicitly.
- Keep common prediction and inference routines in `_ddml_engine.py`. Move the identical clustered-summary and GATE projection helpers there, while leaving ATT's weighted-score-specific GATE function in `_att_impl.py`.
- Move the identical lazy model-bundle class into `_checkpoint_io.py`. Keep `CheckpointStore` in each estimand module because legacy checkpoint lookup and provenance are estimand-specific. Re-export moved helper names from the estimand modules where practical to preserve current imports.

Keep the large estimation, sensitivity, and GATE coordinators in their current estimand files for this first pass. Do not introduce a base class, plugin system, or a single configurable ATE/ATT mega-workflow. After this cleanup, review the remaining duplication before deciding whether a second extraction would reduce complexity or merely add indirection.

## Behavior and reproducibility requirements

- Preserve complete-case rows, control columns and their order, treatment levels, ATT target weights, learner settings, seeds, fold assignments, scores, clustered inference, output names, checkpoint serialization, and run order.
- Preserve the current command-line entry points and their defaults.
- Add each new helper file to the relevant checkpoint provenance inputs. Do not weaken source-code provenance or silently reuse checkpoints under a changed fingerprint.
- Because the source file set and contents change, the new fingerprint is expected to differ. Existing checkpoint files remain untouched, but a later run will not treat them as current automatically. Do not launch a full estimation run as part of the refactor.

## Validation

Use focused unit tests for shared data preparation, deterministic folds, PSU isolation, checkpoint bundle loading, and clustered inference/GATE output shape. Retain estimand-specific tests for ATE scores, ATT weights, and outputs. Run the project Python unit-test command after implementation. Do not run the full ATE/ATT pipeline as a routine check because it is expensive and writes research outputs.

## Out of scope

- Changing the treatment definitions, estimands, controls, learners, folds, standard errors, or sensitivity formulas.
- Changing R/Stata code, raw data, generated tables, or existing checkpoints.
- Renaming the numbered entry points or changing the default clustered run.
- A broad formatting-only rewrite or an attempt to force every duplicate function into a shared abstraction.
