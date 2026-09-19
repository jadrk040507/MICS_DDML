# GATE tables: source-water deciles and three risk groups

Run from the project root without refitting nuisance models:

```bash
"Do file/Python/.venv/bin/python" "Do file/Python/generate_gate_tables.py"
```

Both `ate.py` and `att.py` also generate these tables during their regular GATE stage. The standalone script retains existing GATE results and computes missing groupings from saved model scores. Results aggregate cross-fitting repetitions using the corresponding existing ATE/ATT GATE routines.

## Main tables

| Estimand | Source-water deciles | Three source-risk groups |
|---|---|---|
| ATE: water quality | [Deciles](ATE/table_gate_main_deciles_ecoli.tex) | [Risk groups](ATE/table_gate_main_risk_groups_ecoli.tex) |
| ATE: diarrhea | [Deciles](ATE/table_gate_main_deciles_diarrhea.tex) | [Risk groups](ATE/table_gate_main_risk_groups_diarrhea.tex) |
| ATT: water quality | [Deciles](ATT/table_gate_main_deciles_ecoli.tex) | [Risk groups](ATT/table_gate_main_risk_groups_ecoli.tex) |
| ATT: diarrhea | [Deciles](ATT/table_gate_main_deciles_diarrhea.tex) | [Risk groups](ATT/table_gate_main_risk_groups_diarrhea.tex) |

Main tables use clustered folds. Corresponding `table_gate_appendix_deciles_*.tex` and `table_gate_appendix_risk_groups_*.tex` include clustered and ordinary-fold specifications. Existing `table_gate_main_*.tex` and `table_gate_appendix_*.tex` files that combine both groupings are retained. All filenames, captions and notes are in English. LaTeX requires booktabs, pdflscape and adjustbox.

## Variable provenance

The three groups use the pre-existing `RiskSource` variable in both final Stata datasets. Python does not create new thresholds or recode it:

- 0: No Risk Source, 0 CFU/100 mL.
- 1: Some Risk Source, 1-100 CFU/100 mL (excludes the very-high-risk category).
- 2: Very High Risk Source, above 100 CFU/100 mL. The stored WQ27 value is top-coded at 101.

`Do file/Stata/1. Cleaning.do` already uses RiskSource values 0, 1 and 2 for stratified summaries. This confirms the grouping predates these Python tables; the original upstream creation of RiskSource was not located in the inspected cleaning script.

Deciles use the pre-existing `wq27_decile` variable, defined by `xtile wq27_decile = WQ27, nq(10)` in that cleaning script. Only labels 1, 5, 6, 7 and 8 are observed in the saved analysis results. The tables preserve observed categories and do not invent empty deciles.

These are source-water groups, not groups defined by the household-water outcome. ATT GATEs retain the original target of users of any treatment within each group. No ATU GATE is derived by this script.
