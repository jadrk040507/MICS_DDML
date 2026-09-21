# Active Python workflow

| Script | Purpose | Output |
|---|---|---|
| `run_analysis.py` | Single entry point for ATE, ATT, or both | See the selected estimand |
| `ate.py` | ATE estimation, sensitivity and both GATE groupings | `Output/ATE/` |
| `att.py` | ATT estimation, sensitivity and both GATE groupings | `Output/ATT/` |
| `ddml_engine.py` | Shared Super Learner prediction engine | No direct output |
| `compare_ate_att_atu.py` | Aggregated ATE/ATT/derived-ATU comparisons and PDF | `Output/ATE_ATT_ATU/` |
| `generate_gate_tables.py` | Decile and three-risk-group GATE tables from saved models | `Output/ATE/`, `Output/ATT/` |
| `compare_ate_att.py` | Supplementary GATE, sensitivity and learner-weight comparisons | `Output/ATE_vs_ATT/` |
| `characteristics.py` | Treatment predictors / sample characteristics | `Output/` |

## Run the estimation

From this directory, after `uv sync`. The project metadata and environment are
stored at the repository root, so `uv run` uses `MICS_DDML/.venv` even when
the command is launched here:

```bash
uv run python run_analysis.py
uv run python run_analysis.py ate
uv run python run_analysis.py att
uv run python run_analysis.py both
```

Running without an argument is equivalent to `both`: ATE runs first and ATT
runs second. The original commands `uv run python ate.py`
and `uv run python att.py` remain valid. The estimand scripts retain their own
sampling and reporting choices; `run_analysis.py` only coordinates them.

## Refresh tables without refitting

From this directory, after `uv sync`:

```bash
uv run python generate_gate_tables.py
uv run python compare_ate_att_atu.py
uv run python compare_ate_att.py
```

Tables and filenames are in English. Comparisons aggregate all repetitions; individual-repetition tables are not generated. Both decile and three-risk-group GATE tables are retained.

## Regression tests

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

`tests/` contains all retained inference and GATE checks. The environment is
rebuilt from the root `pyproject.toml` and `uv.lock`; the root `.venv/` is local
and untracked. Running `ate.py` or `att.py` can fit missing models; use the
table commands above for presentation updates only.

## Checkpoint provenance

ATE and ATT checkpoint filenames contain a 12-character provenance fingerprint. The fingerprint records the estimation script, locked Python environment, research-data inputs, estimand, sample settings, folds, repetitions, treatment definition and learner library. A change to any of these inputs produces a new filename instead of silently loading an incompatible fitted model.

Each `manifest.json` records the complete fingerprint inputs and lists only checkpoints compatible with that run. Older unversioned checkpoints are retained as historical files but are not loaded automatically.

## Archive

`archive/` retains earlier notebooks and implementations. `archive/cleanup_2026-09-09/` contains unused helper modules, old logs and generated caches. These helper modules are not imported by the active workflow. Archived code is historical material, not guaranteed runnable from its new location.

No research data or full-run model checkpoints were deleted. `Output/archive/cleanup_2026-09-09/moves.json` records reversible source/destination paths relative to the project root. Active script locations were preserved to keep imports and project-root discovery stable. Stata preprocessing and analysis scripts remain in `Do file/Stata/`.
