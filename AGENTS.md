# MICS_DDML project guidance

This repository contains a research pipeline implemented in Python and Stata.

## Working agreements

- Use the project skill `.agents/skills/research-code-audit/SKILL.md` for code reviews, replication checks, and pipeline audits.
- Review Python before Stata unless the task explicitly requests another order.
- Begin reviews with read-only inspection. Do not modify analysis code, source data, checkpoints, or generated results unless the user explicitly asks for fixes or regeneration.
- For Python, inspect the declared environment and existing tests before running checks. Prefer the locked project environment and repository commands.
- For Stata, preserve macro scope, sort order, merge keys, missing-value semantics, weights, clustering, and version compatibility. Do not execute `.do` files unless Stata and the required dependencies are available and execution is explicitly in scope.
- Treat `Data/1. Raw/` as immutable. Write generated results only to the intended clean/final/output locations.
- Report findings in impact order with exact file and line evidence. Separate confirmed defects from unverified risks.

## Canonical commands

The canonical environment is `.venv` at the repository root. Run Python
commands from `Do file/Python`; `uv` will discover the root `pyproject.toml`
and use that environment:

```bash
uv sync
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python run_analysis.py ate
uv run python run_analysis.py att
uv run python run_analysis.py
```

The final command runs ATE followed by ATT. Full estimation may be expensive
and may write checkpoints and publication outputs, so do not run it as a
routine validation step unless execution is explicitly in scope.
