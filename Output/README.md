# Current analysis outputs

| Folder | Contents |
|---|---|
| [ATE](ATE/) | Full-run ATE checkpoints, estimates, sensitivity and GATE tables |
| [ATT](ATT/) | Full-run ATT checkpoints, estimates, sensitivity and GATE tables |
| [ATE_ATT_ATU](ATE_ATT_ATU/README.md) | Aggregated comparison, joint inference, PDF and numeric inputs |
| [ATE_vs_ATT](ATE_vs_ATT/README.md) | Supplementary GATE, sensitivity and learner-weight comparisons |
| [archive](archive/cleanup_2026-09-09/moves.json) | Historical sample/smoke/stale outputs and editor previews |

Start with the [comparison PDF](ATE_ATT_ATU/comparison_ate_att_atu.pdf). The [GATE index](GATE_README.md) links decile and three-risk-group tables for both estimands.

## Required files

- `checkpoints/`: fitted full-run models and sensitivity checkpoints; needed to avoid refitting.
- `results_*.pkl` and `joint_covariance*.pkl`: estimates, descriptive statistics and joint uncertainty.
- `.tex`, `.md` and the comparison PDF: current presentation outputs.
- `manifest.json`: original run provenance. These historical records may list sample checkpoints now archived.

Repetition-level numeric inputs are retained for aggregate inference, but individual-repetition tables are not generated.

## Cleanup record

Old logs, unused Python helpers, generated caches, temporary previews and files explicitly tagged `sample05`, `smoke` or `stale` were moved to archives. Nothing was permanently deleted. Full-run checkpoint and table paths are unchanged. A future sample-only run may regenerate its archived caches.

The [move log](archive/cleanup_2026-09-09/moves.json) records each source, destination and reason. See [active scripts and commands](../Do%20file/Python/README.md).
