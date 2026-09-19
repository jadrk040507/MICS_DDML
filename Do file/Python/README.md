# Active Python workflow

| Script | Purpose | Output |
|---|---|---|
| `ate.py` | ATE estimation, sensitivity and both GATE groupings | `Output/ATE/` |
| `att.py` | ATT estimation, sensitivity and both GATE groupings | `Output/ATT/` |
| `compare_ate_att_atu.py` | Aggregated ATE/ATT/derived-ATU comparisons and PDF | `Output/ATE_ATT_ATU/` |
| `generate_gate_tables.py` | Decile and three-risk-group GATE tables from saved models | `Output/ATE/`, `Output/ATT/` |
| `compare_ate_att.py` | Supplementary GATE, sensitivity and learner-weight comparisons | `Output/ATE_vs_ATT/` |
| `characteristics.py` | Treatment predictors / sample characteristics | `Output/` |

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

`tests/` contains all retained inference and GATE checks. The environment is rebuilt from `pyproject.toml` and `uv.lock`; `.venv/` is local and untracked. Running `ate.py` or `att.py` can fit missing models; use the table commands above for presentation updates only.

## Archive

`archive/` retains earlier notebooks and implementations. `archive/cleanup_2026-09-09/` contains unused helper modules, old logs and generated caches. These helper modules are not imported by the six active scripts. Archived code is historical material, not guaranteed runnable from its new location.

No research data or full-run model checkpoints were deleted. `Output/archive/cleanup_2026-09-09/moves.json` records reversible source/destination paths relative to the project root. Active script locations were preserved to keep imports and project-root discovery stable. Stata preprocessing and analysis scripts remain in `Do file/Stata/`.
