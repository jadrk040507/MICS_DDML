# MICS water treatment and child health

This repository contains the active Python and Stata research workflow for studying household water treatment, microbial water quality, and child diarrhea with UNICEF Multiple Indicator Cluster Surveys (MICS).

The main estimators use Double/Debiased Machine Learning with cross-fitting. The code distinguishes average treatment effects (ATE), effects for treated households (ATT), and derived effects for untreated households (ATU). It also reports heterogeneity by source-water risk and compares clustered with observation-level inference.

## Active workflow

- `Do file/Python/run_analysis.py` is the canonical entry point for running
  ATE, ATT, or both estimands.
- `Do file/Python/ate.py` estimates ATE specifications.
- `Do file/Python/att.py` estimates ATT specifications.
- `Do file/Python/ddml_engine.py` contains the prediction and inference
  machinery shared by ATE and ATT.
- `Do file/Python/provenance.py` fingerprints data, code, settings, and the
  locked environment so stale checkpoints are not loaded silently.
- `Do file/Python/compare_ate_att_atu.py` builds joint ATE/ATT/ATU comparisons.
- `Do file/Python/generate_gate_tables.py` regenerates heterogeneity tables from saved models.
- `Do file/Python/tests/` contains inference and aggregation regression tests.
- `Do file/Stata/` contains data preparation, descriptive analysis, and the Stata DDML specification.
- `Writing edit/` contains the paper source and publication figures.
- `Output/` contains lightweight manifests and publication tables. Fitted model objects and checkpoints are intentionally excluded.

More detailed Python instructions are in `Do file/Python/README.md`.

## Data and reproducibility

MICS microdata are not distributed here. Authorized users must obtain the source files under UNICEF's terms and place them under `Data/`, which is ignored by Git. The Dropbox working directory may also contain reference PDFs, questionnaires, local environments, and model checkpoints; these are excluded from the public repository.

Create the Python environment from the lockfile. The `pyproject.toml`,
`uv.lock`, `.python-version`, and canonical `.venv` all live at the repository
root. Commands run from `Do file/Python` discover that root project
automatically:

```bash
cd "Do file/Python"
uv sync
```

For VS Code or Spyder, select `MICS_DDML/.venv/bin/python` as the interpreter.
There is no second environment under `Do file/Python`.

Run the complete analysis:

```bash
uv run python run_analysis.py
```

The default runs ATE first and ATT second. To run only one estimand:

```bash
uv run python run_analysis.py ate
uv run python run_analysis.py att
```

Full estimation requires the authorized datasets and can be computationally
expensive. Existing checkpoints are reused only when their provenance
fingerprint matches the current data, code, configuration, and lockfile.

Run the regression tests:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Presentation tables can be regenerated from compatible saved checkpoints
using the commands documented in the Python README.

## Interpretation

The data are observational. Causal interpretation depends on conditional exchangeability, overlap, consistency, correct sample construction, and the validity of the clustering design. Machine-learning adjustment and orthogonal scores reduce specific estimation biases but do not rule out unobserved confounding. Numerical results should therefore be read together with uncertainty intervals, sensitivity analyses, and the exact sample definition.

## Author

Juan Alvaro Díaz Raimond Kedilhac — [GitHub](https://github.com/jadrk040507) · [LinkedIn](https://www.linkedin.com/in/jadrk040507/)
