# Consolidated Python Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated ATE/ATT Python pipeline with one runner and five focused active modules while preserving the statistical analysis and current outputs.

**Architecture:** `run_analysis.py` expands CLI choices and calls one workflow in `analysis.py`. Statistical mechanics live in `ddml.py`, publication and stage outputs in `reporting.py`, and current-fingerprint persistence in `artifacts.py`; ATE and ATT differences are named configuration and small strategy functions rather than separate implementations.

**Tech Stack:** Python 3.14, pandas, NumPy, scikit-learn, DoubleML, joblib, XGBoost, `argparse`, `unittest`, and the locked `uv` environment.

**Spec:** `docs/superpowers/specs/2026-09-27-consolidated-python-workflow-design.md`

## Global Constraints

- Keep the default command equivalent to clustered folds, both estimands, and all stages in ATE-before-ATT order.
- Preserve household controls, child age/sex additions, complete-case rows, categorical reference levels, encoded-column order, treatment levels `(0, 1, 2, 3, 98)`, and reported levels `(0, 1, 2, 3)`.
- Preserve seeds, fold counts, repetition seeds, PSU isolation, learner definitions, ATT target population, propensity clipping, ATT weights, score signs, influence-array dimensions, clustered covariance, and bootstrap settings.
- Preserve current result filenames, checkpoint payload shapes, and the `Output/ATE/` and `Output/ATT/` directories.
- Remove legacy checkpoint discovery and compatibility aliases. Leave existing checkpoint files untouched; the new source fingerprint intentionally prevents automatic reuse.
- Remove the numbered runners, historical ATE/ATT/ATU report, its joint-inference helper, obsolete wrappers, and `Do file/Python/archive/`.
- Do not modify or execute Stata, raw/final data, existing checkpoints, or generated research outputs.
- Do not launch a full ATE/ATT estimation as part of validation.

## Review Focus

1. A clustered sample with a missing PSU must drop that row, while a missing optional metadata value must not; pin this in Task 1.
2. A corrupt current-fingerprint checkpoint must raise a useful load error rather than fall back to a legacy file or silently look current; pin this in Task 2.
3. Reporting for an estimand/outcome with only one requested fold mode must not access the absent mode's model bundle; pin this in Task 3.
4. ATT and ATE must call the same orchestration function while retaining different score/weight strategies; pin this in Task 4.
5. Invalid CLI values and an import from a clean checkout must fail cleanly or succeed without any removed module; pin this in Tasks 5 and 6.

---

### Task 1: Consolidate data, cross-fitting, and statistical machinery

**Files:**
- Create: `Do file/Python/ddml.py`
- Modify: `Do file/Python/_ate_impl.py`
- Modify: `Do file/Python/_att_impl.py`
- Delete: `Do file/Python/_analysis_data.py`
- Delete: `Do file/Python/_cross_fitting.py`
- Delete: `Do file/Python/_ddml_engine.py`
- Modify: `Do file/Python/tests/test_analysis_data.py`
- Modify: `Do file/Python/tests/test_cross_fitting.py`
- Modify: `Do file/Python/tests/test_common_engine.py`
- Modify: `Do file/Python/tests/test_aggregate_effects.py`
- Modify: `Do file/Python/tests/test_compact_progress.py`

**Interfaces:**
- Produces: `controls_for_sample(common_controls, child_controls, child) -> list[str]`.
- Produces: `complete_case_sample(data, outcome, treatment, controls, *, cluster=True, cluster_column="Cluster_var", country_column="country_cat", allowed_levels=None, extra_columns=()) -> pandas.DataFrame`.
- Produces: `make_frame(data, outcome, treatment, controls, *, categorical_controls, cluster=True, cluster_column="Cluster_var", country_column="country_cat", allowed_levels=None) -> tuple[pandas.DataFrame, list[str]]`.
- Produces: `load_analysis_data(path, outcome, controls, *, country_codes=None, quick_sample=False, sample_fraction=0.05, sample_seed=42) -> pandas.DataFrame`.
- Produces the existing `ConvexRegressor`, `ConvexClassifier`, fold, score, clustered-inference, sensitivity-framework, common-GATE, and formatting interfaces under `ddml`.
- Preserves the temporary `_cluster_model_code` contract: it is the last nuisance column and never enters a base learner as a predictor.

- [ ] **Step 1: Add failing consolidation and missing-value tests**

Add `test_ddml_exports_data_fold_and_inference_api`, `test_clustered_sample_drops_missing_psu`, and `test_optional_metadata_does_not_drop_rows`. Update existing imports to `ddml` only after watching the export test fail because `ddml.py` is absent.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_analysis_data.py' && uv run python -m unittest discover -s tests -p 'test_common_engine.py'`

Expected: FAIL on missing `ddml` API; the two row-selection tests must be capable of distinguishing required PSU fields from optional metadata.

- [ ] **Step 3: Build `ddml.py` and switch every consumer**

Move the live implementations without changing their bodies or defaults. Replace imports in both estimand modules and all affected tests. Remove the three superseded helper modules only after `rg` shows no remaining imports.

- [ ] **Step 4: Run all statistical consumers**

Run: `uv run python -m unittest discover -s tests -p 'test_analysis_data.py' && uv run python -m unittest discover -s tests -p 'test_cross_fitting.py' && uv run python -m unittest discover -s tests -p 'test_common_engine.py' && uv run python -m unittest discover -s tests -p 'test_aggregate_effects.py' && uv run python -m unittest discover -s tests -p 'test_compact_progress.py'`

Expected: all tests pass; ordinary splits remain deterministic and clustered splits contain no PSU overlap.

- [ ] **Step 5: Commit**

```bash
git add 'Do file/Python/ddml.py' 'Do file/Python/_ate_impl.py' 'Do file/Python/_att_impl.py' 'Do file/Python/tests'
git add -u -- 'Do file/Python/_analysis_data.py' 'Do file/Python/_cross_fitting.py' 'Do file/Python/_ddml_engine.py'
git commit -m "Consolidate DDML statistical machinery"
```

### Task 2: Consolidate current checkpoint and provenance handling

**Files:**
- Create: `Do file/Python/artifacts.py`
- Modify: `Do file/Python/_ate_impl.py`
- Modify: `Do file/Python/_att_impl.py`
- Delete: `Do file/Python/_checkpoint_io.py`
- Delete: `Do file/Python/_provenance.py`
- Delete: `Do file/Python/_model_checkpoint_compat.py`
- Modify: `Do file/Python/tests/test_checkpoint_bundle.py`
- Modify: `Do file/Python/tests/test_provenance.py`
- Replace: `Do file/Python/tests/test_model_checkpoint_compat.py` with `Do file/Python/tests/test_artifacts.py`

**Interfaces:**
- Consumes: current helper paths from Task 1 when building model provenance.
- Produces: `build_checkpoint_provenance(schema_version, files, settings) -> tuple[str, dict]`.
- Produces: `build_sensitivity_provenance(schema_version, model_fingerprint, files, settings=None) -> tuple[str, dict]`.
- Produces: `atomic_dump(value, path) -> None`, `valid_sensitivity_rows(...) -> bool`, and `OutcomeCheckpointBundle(paths, table_frames)` with the existing payload-unwrapping behavior.
- Produces: `CheckpointStore(checkpoint_dir, *, estimand, quick_sample, model_fingerprint, sensitivity_fingerprint, sample_fraction=0.05)` with `path`, `exists`, `load`, and `save` methods.
- `CheckpointStore.load` reads only its fingerprinted `path(name)` and propagates joblib deserialization errors; it never searches legacy directories.

- [ ] **Step 1: Add failing current-only artifact tests**

Add `test_artifacts_exports_current_checkpoint_api`, `test_missing_current_checkpoint_does_not_search_legacy_paths`, `test_corrupt_current_checkpoint_raises_load_error`, and retain the lazy APOS payload tests.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_artifacts.py' && uv run python -m unittest discover -s tests -p 'test_provenance.py'`

Expected: FAIL because `artifacts` and the generic `CheckpointStore` do not exist.

- [ ] **Step 3: Implement current-only artifacts and remove compatibility**

Move atomic writes, validation, bundle loading, and provenance into `artifacts.py`. Parameterize the duplicated checkpoint stores by output directory and estimand. Delete `legacy_model_path`, `sys.modules.setdefault(...)`, and every old-name compatibility import.

- [ ] **Step 4: Run checkpoint, provenance, and estimand tests**

Run: `uv run python -m unittest discover -s tests -p 'test_artifacts.py' && uv run python -m unittest discover -s tests -p 'test_checkpoint_bundle.py' && uv run python -m unittest discover -s tests -p 'test_provenance.py' && uv run python -m unittest discover -s tests -p 'test_ate_inference.py'`

Expected: all pass; fingerprints include `analysis.py` when it exists later, plus `ddml.py`, `reporting.py` when it exists later, and `artifacts.py`. Until Tasks 3–4 create those files, absent future paths must not be added as `missing` records.

- [ ] **Step 5: Commit**

```bash
git add 'Do file/Python/artifacts.py' 'Do file/Python/_ate_impl.py' 'Do file/Python/_att_impl.py' 'Do file/Python/tests'
git add -u -- 'Do file/Python/_checkpoint_io.py' 'Do file/Python/_provenance.py' 'Do file/Python/_model_checkpoint_compat.py'
git commit -m "Use current-only checkpoint artifacts"
```

### Task 3: Extract shared reporting, sensitivity, and GATE stages

**Files:**
- Create: `Do file/Python/reporting.py`
- Modify: `Do file/Python/_ate_impl.py`
- Modify: `Do file/Python/_att_impl.py`
- Delete: `Do file/Python/_sensitivity_groups.py`
- Delete: `Do file/Python/_sensitivity_scale.py`
- Create: `Do file/Python/tests/test_reporting.py`
- Modify: `Do file/Python/tests/test_gate_groups.py`
- Modify: `Do file/Python/tests/test_sensitivity_scale.py`

**Interfaces:**
- Consumes: `AnalysisSpec`-like objects with `estimand`, `output_dir`, `reported_levels`, and `att_gate_strategy`; use a `typing.Protocol` in `reporting.py` so it does not import `analysis.py` at runtime.
- Produces: `save_effect_outputs(spec, estimates, *, quick_sample, file_suffix="", fold_mode="both") -> None`.
- Produces: `run_sensitivity(spec, estimates, *, quick_sample, fold_mode="both") -> pandas.DataFrame`.
- Produces: `run_gate(spec, estimates, *, quick_sample, fold_mode="both") -> pandas.DataFrame`.
- Produces: `write_manifest(spec, *, model_provenance, sensitivity_provenance, fold_mode) -> pathlib.Path`.
- Absorbs benchmark group definitions and diagonal-equivalent sensitivity formulas.

- [ ] **Step 1: Add failing shared-reporting tests**

Add `test_reporting_exports_all_three_stage_functions`, `test_single_fold_mode_never_requests_absent_bundle`, and fixture comparisons asserting ATE and ATT coefficient, SE, interval, `n`, and `n_psu` rows remain unchanged.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_reporting.py' && uv run python -m unittest discover -s tests -p 'test_gate_groups.py'`

Expected: FAIL because `reporting.py` does not exist.

- [ ] **Step 3: Move shared output stages and parameterize real differences**

Move the exact duplicate table, sensitivity, E. coli range, and GATE orchestration functions once. Pass estimand labels and ATT-specific GATE calculation explicitly through `spec`; do not branch on the importing module name. Remove empty section headings from both temporary estimand modules.

- [ ] **Step 4: Run all reporting consumers**

Run: `uv run python -m unittest discover -s tests -p 'test_reporting.py' && uv run python -m unittest discover -s tests -p 'test_gate_groups.py' && uv run python -m unittest discover -s tests -p 'test_sensitivity_scale.py' && uv run python -m unittest discover -s tests -p 'test_ate_inference.py' && uv run python -m unittest discover -s tests -p 'test_att.py'`

Expected: all pass for clustered and unclustered fixtures and for ATE and ATT result rows.

- [ ] **Step 5: Commit**

```bash
git add 'Do file/Python/reporting.py' 'Do file/Python/_ate_impl.py' 'Do file/Python/_att_impl.py' 'Do file/Python/tests'
git add -u -- 'Do file/Python/_sensitivity_groups.py' 'Do file/Python/_sensitivity_scale.py'
git commit -m "Share reporting and robustness stages"
```

### Task 4: Replace separate ATE and ATT implementations with one workflow

**Files:**
- Create: `Do file/Python/analysis.py`
- Delete: `Do file/Python/_ate_impl.py`
- Delete: `Do file/Python/_att_impl.py`
- Create: `Do file/Python/tests/test_analysis_workflow.py`
- Modify: `Do file/Python/tests/test_att.py`
- Modify: `Do file/Python/tests/test_ate_inference.py`
- Modify: `Do file/Python/tests/test_learner_spec.py`
- Modify: `Do file/Python/tests/test_common_engine.py`
- Modify: `Do file/Python/tests/test_provenance.py`

**Interfaces:**
- Consumes: `ddml`, `artifacts`, and `reporting` interfaces from Tasks 1–3.
- Produces: frozen `AnalysisSpec` with named fields for `estimand`, `output_dir`, `score`, `target_levels`, `propensity_clip`, `uses_att_weights`, and `att_gate_strategy`.
- Produces: `get_analysis_spec(estimand: str) -> AnalysisSpec`; accepted values are `"ate"` and `"att"`.
- Produces: `estimate_one_outcome(spec, dataset, data_path, child, outcome, country_codes, checkpoint_prefix, quick_sample, fold_mode="both") -> dict`.
- Produces: `estimate_all_models(spec, country_codes, checkpoint_prefix, quick_sample, fold_mode="both") -> dict`.
- Produces: `run_analysis(spec, *, fold_mode="clustered", stage="all") -> None`.
- Keeps named ATT-only functions `make_att_weights(...)` and `estimate_att_gate_from_scores(...)`; ATE does not call them.

- [ ] **Step 1: Add failing shared-workflow tests**

Add `test_ate_and_att_use_same_estimation_function`, `test_att_spec_selects_atte_and_weight_strategy`, `test_ate_spec_does_not_construct_att_weights`, and `test_analysis_rejects_unknown_estimand`.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_analysis_workflow.py'`

Expected: FAIL because `analysis.py` and `AnalysisSpec` do not exist.

- [ ] **Step 3: Implement `analysis.py` from the remaining estimand modules**

Move shared constants, learner definitions, data wrappers, fit strategies, selected-country workflow, and stage orchestration once. Replace estimand-dependent branches with named `AnalysisSpec` values or the two explicit ATT strategy functions. Remove duplicate section numbering; use short headings only where a reader needs navigation.

- [ ] **Step 4: Delete both old estimand modules and update all imports**

Run `rg '_ate_impl|_att_impl' 'Do file/Python' -g '*.py' -g '!archive/**'`; update every active consumer, then delete both files. The search must return no active-code match.

- [ ] **Step 5: Run analysis, ATT, learner, inference, and provenance tests**

Run: `uv run python -m unittest discover -s tests -p 'test_analysis_workflow.py' && uv run python -m unittest discover -s tests -p 'test_att.py' && uv run python -m unittest discover -s tests -p 'test_ate_inference.py' && uv run python -m unittest discover -s tests -p 'test_learner_spec.py' && uv run python -m unittest discover -s tests -p 'test_common_engine.py' && uv run python -m unittest discover -s tests -p 'test_provenance.py'`

Expected: all pass; provenance now records exactly `analysis.py`, `ddml.py`, `reporting.py`, and `artifacts.py` as code inputs.

- [ ] **Step 6: Commit**

```bash
git add 'Do file/Python/analysis.py' 'Do file/Python/tests'
git add -u -- 'Do file/Python/_ate_impl.py' 'Do file/Python/_att_impl.py'
git commit -m "Run ATE and ATT through one analysis workflow"
```

### Task 5: Replace numbered scripts with the single CLI

**Files:**
- Create: `Do file/Python/run_analysis.py`
- Delete: `Do file/Python/01_run_analysis.py`
- Delete: `Do file/Python/02_ate_c.py`
- Delete: `Do file/Python/03_att_c.py`
- Delete: `Do file/Python/04_sensitivity_c.py`
- Delete: `Do file/Python/05_gate_c.py`
- Delete: `Do file/Python/06_ate_u.py`
- Delete: `Do file/Python/07_att_u.py`
- Delete: `Do file/Python/08_sensitivity_u.py`
- Delete: `Do file/Python/09_gate_u.py`
- Delete: `Do file/Python/_analysis_runner.py`
- Modify: `Do file/Python/tests/test_run_analysis_modes.py`

**Interfaces:**
- Consumes: `analysis.get_analysis_spec` and `analysis.run_analysis` from Task 4.
- Produces: `run(fold_mode="clustered", estimand="both", stage="all") -> None`.
- Produces: `main(argv=None) -> None` with positional fold mode choices `clustered|unclustered|all`, `--estimand ate|att|both`, and `--stage effects|sensitivity|gate|all`.
- Default expansion is `(clustered, ate, all)` followed by `(clustered, att, all)` through the stage order effects, sensitivity, gate.

- [ ] **Step 1: Rewrite runner tests against the new CLI and watch RED**

Add `test_default_runs_clustered_ate_then_att_in_stage_order`, `test_all_expands_clustered_before_unclustered`, `test_estimand_and_stage_filters`, and `test_invalid_cli_value_exits_with_argparse_error`.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_run_analysis_modes.py'`

Expected: FAIL because the new `run_analysis.py` interface is absent.

- [ ] **Step 3: Implement the single runner and delete wrappers**

Restore NumPy random state after each `(estimand, fold mode, stage)` execution. Delete the ten superseded runner files only after every test imports `run_analysis` directly.

- [ ] **Step 4: Run runner tests and syntax checks**

Run: `uv run python -m unittest discover -s tests -p 'test_run_analysis_modes.py' && uv run python -m py_compile run_analysis.py analysis.py ddml.py reporting.py artifacts.py`

Expected: runner tests pass and all five active modules compile.

- [ ] **Step 5: Commit**

```bash
git add 'Do file/Python/run_analysis.py' 'Do file/Python/tests/test_run_analysis_modes.py'
git add -u -- 'Do file/Python/01_run_analysis.py' 'Do file/Python/02_ate_c.py' 'Do file/Python/03_att_c.py' 'Do file/Python/04_sensitivity_c.py' 'Do file/Python/05_gate_c.py' 'Do file/Python/06_ate_u.py' 'Do file/Python/07_att_u.py' 'Do file/Python/08_sensitivity_u.py' 'Do file/Python/09_gate_u.py' 'Do file/Python/_analysis_runner.py'
git commit -m "Expose one analysis command"
```

### Task 6: Remove historical code and enforce the final file boundary

**Files:**
- Delete: `Do file/Python/_compare_ate_att_atu.py`
- Delete: `Do file/Python/_joint_inference.py`
- Delete: `Do file/Python/__init__.py`
- Delete: `Do file/Python/archive/`
- Delete: `Do file/Python/tests/test_ate_att_atu.py`
- Modify: `Do file/Python/tests/test_aggregate_effects.py`
- Modify: `Do file/Python/tests/test_repository_layout.py`
- Modify: `Do file/Python/README.md`

**Interfaces:**
- Consumes: the five final active modules from Tasks 1–5.
- Produces: an exact root implementation set of `run_analysis.py`, `analysis.py`, `ddml.py`, `reporting.py`, and `artifacts.py` excluding `tests/`.
- Produces README commands matching the Task 5 CLI and documents the one-time checkpoint fingerprint change.

- [ ] **Step 1: Add failing final-layout and removed-import tests**

Replace the current tracked-file test with `test_python_root_contains_only_five_active_modules`, `test_no_active_import_mentions_removed_modules`, and `test_clean_import_of_all_active_modules`. The exact root `.py` set must equal the five documented names.

- [ ] **Step 2: Run the RED tests**

Run: `uv run python -m unittest discover -s tests -p 'test_repository_layout.py'`

Expected: FAIL while historical modules, numbered wrappers, or `archive/` remain.

- [ ] **Step 3: Delete historical code and update remaining consumers**

Remove the historical comparison and archive. Move any still-used aggregate-repetition assertion into a current inference fixture; do not preserve a production helper solely for its old test.

- [ ] **Step 4: Rewrite the README around the single command and five-file map**

Document defaults, examples for estimand/fold/stage filters, output directories, fingerprint consequences, and the absence of legacy checkpoint loading. Remove numbered-script and historical-report references.

- [ ] **Step 5: Run layout, import, and documentation-facing tests**

Run: `uv run python -m unittest discover -s tests -p 'test_repository_layout.py' && uv run python -m unittest discover -s tests -p 'test_run_analysis_modes.py' && uv run python -m unittest discover -s tests -p 'test_provenance.py'`

Expected: all pass; `rg '_ate_impl|_att_impl|_analysis_runner|_model_checkpoint_compat|_compare_ate_att_atu|_joint_inference' 'Do file/Python' -g '*.py'` returns no match.

- [ ] **Step 6: Commit**

```bash
git add 'Do file/Python/README.md' 'Do file/Python/tests'
git add -u -- 'Do file/Python/__init__.py' 'Do file/Python/_compare_ate_att_atu.py' 'Do file/Python/_joint_inference.py' 'Do file/Python/archive'
git commit -m "Remove superseded Python workflow files"
```

### Task 7: Full clean-checkout regression and scope audit

**Files:**
- Review only: `Do file/Python/`, `Data/`, `Output/`, `Do file/Stata/`, `Writing edit/`

**Interfaces:**
- Consumes: the complete five-module workflow.
- Produces: verification evidence only; no production code or output changes.

- [ ] **Step 1: Run the complete suite in the working branch**

Run from `Do file/Python`: `uv run python -m unittest discover -s tests -p 'test_*.py'`

Expected: all discovered tests pass with zero failures and zero errors.

- [ ] **Step 2: Verify imports, whitespace, and blast radius**

Run: `git diff --check <implementation-base>..HEAD`, compile all five modules, and use `rg` to confirm every removed module name has zero active Python consumer. Inspect the diff for controls, levels, seeds, output filenames, checkpoint payload keys, ATT scores, and runner defaults.

- [ ] **Step 3: Run the suite from a detached clean worktree**

Create a temporary detached worktree at `HEAD`, run the canonical suite there with the repository `.venv`, and remove the temporary worktree.

Expected: the same test count passes without access to untracked workspace files.

- [ ] **Step 4: Confirm protected paths are outside the commit range**

Run: `git diff --name-only <implementation-base>..HEAD` and assert there are no paths under `Data/`, `Output/`, `Do file/Stata/`, or `Writing edit/`.

- [ ] **Step 5: Confirm no expensive run occurred**

Record that validation invoked only unit tests, compilation, imports, and read-only Git checks; no command called `run_analysis.py` without mocks and no existing checkpoint or generated result changed.
