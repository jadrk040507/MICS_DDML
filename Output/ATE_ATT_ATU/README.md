# ATE, ATT and derived ATU

## Read the tables

- [Publication PDF](comparison_ate_att_atu.pdf): aggregated main comparisons and ATT-minus-ATU contrasts only.
- [Full sample, clustered folds](comparison_clustered.md).
- [Selected countries, clustered folds](comparison_selected_countries_clustered.md).
- [Full sample, ordinary folds](comparison_iid.md).
- [Selected countries, ordinary folds](comparison_selected_countries_iid.md).

All table content is in English. Main tables follow `Output/ATE/table_water_treatment_main.tex` and `Output/ATE_vs_ATT/table_ate_att_main.tex`: outcome column groups, IRM/APOS panels, coefficients with three decimals and significance stars, standard errors in parentheses, descriptive statistics, observations, PSUs and explanatory notes. ATE, ATT and ATU appear side by side under each outcome. The other-treatment contrast is retained for completeness.

**Units:** coefficients and standard errors are in probability units; 0.01 equals one percentage point. Descriptive means and treatment shares are explicitly labeled as percentages. Stars denote pointwise p-values: *** p<0.01, ** p<0.05, * p<0.1. Stars on individual effects do not test differences; use the separate ATT-minus-ATU tables.

## Run

From the project root:

```bash
"Do file/Python/.venv/bin/python" "Do file/Python/compare_ate_att_atu.py"
```

Default: full sample and pooled selected countries (Dominican Republic, Guyana, Honduras and Malawi), all three outcomes, IRM and APOS, clustered and ordinary folds. `--selected-countries` restricts the run to that scope. The script reads trusted project checkpoints and does not refit models. If `pdflatex` is installed, it also compiles the standalone PDF.

## Source and output folders

- `Output/ATE/`: ATE checkpoints, results and original tables.
- `Output/ATT/`: ATT checkpoints, results and original tables.
- `Output/ATE_ATT_ATU/`: joint results and comparison tables.

The paired model data must match exactly, including controls, outcome, treatment, PSU where present and row order. GATE and sensitivity outputs remain in their original estimand folders; this script does not derive ATU GATE or sensitivity estimates. Both source-water decile and risk-group GATE tables are available for ATE and ATT; see ../GATE_README.md.

## Estimands and inference

ATE averages over the analysis sample. ATT targets observations using any treatment; ATU targets observations using none. For U5 these populations are children in treated or untreated households. In APOS the same any-treatment target is used for every method-versus-none contrast; ATT is not restricted to users of that specific method.

Let p be the treated share in the estimation sample:

    ATU = (ATE - p ATT)/(1-p)
    ATE = p ATT + (1-p) ATU

ATU is derived, not independently fitted. The identity holds by construction and does not validate identification assumptions. Estimates and descriptive statistics are unweighted by survey weights. Causal interpretations retain the source models' identification assumptions.

Main tables aggregate all three cross-fitting repetitions using the existing project rule. For each estimand, the coefficient, p-value and confidence-interval endpoints are the medians of their repetition-specific values. The reported SE is (median(coef_rep + 1.96 * se_rep) - median(coef_rep)) / 1.96, matching `ate.cluster_robust_framework_inference`.

ATU and ATT-minus-ATU are calculated within each repetition, with joint covariance, before aggregation. The aggregated ATT-minus-ATU can differ from the difference of the displayed median ATT and ATU. The displayed median ATE, ATT and ATU need not satisfy the exact decomposition. No aggregate decomposition is manufactured. Individual-repetition tables are not published.

Each comparison table contains one aggregated result for its scope and specification. Repetition-level numeric inputs remain in the pickle files for reproducibility, but no individual-repetition tables are generated.

The influence function of ATU includes estimation of p:

    IF_ATU = [IF_ATE - p IF_ATT + (ATU-ATT) IF_p]/(1-p)
    IF_p = D-p

The covariance matrix uses summed influence scores by PSU for clustered specifications and observation-level scores for ordinary specifications, divided by N squared, without a degrees-of-freedom correction. Confidence intervals use normal critical values. The joint order is `(ATE, ATT, ATU, p)`. `linear_combination(theta, covariance, weights)` supports fixed-coefficient contrasts; `[0,1,-1,0]` gives ATT-minus-ATU. The decomposition with estimated p is nonlinear in this joint vector and must not treat p as fixed for inference.

## Files

- `results_ate_att_atu.pkl` and `results_ate_att_atu_selected_countries.pkl`: repetition-level estimates and contributions.
- `results_ate_att_atu_aggregated*.pkl`: aggregated coefficients and inference; decomposition fields are intentionally missing. `n_repetitions` records the number used and `repetition=0` is an internal aggregate sentinel, never a fitted repetition.
- `joint_covariance*.pkl`: joint vectors and covariance matrices.
- `comparison*.tex` / `.md`: publication comparisons and readable companion results; all output filenames are in English.
- `table_contrasts*_{clustered,iid}.tex`: ATT-minus-ATU tests.
- `table_inference*.tex`: detailed aggregated estimates and inference, split into ordinary tables by outcome and model; requires booktabs and adjustbox, not longtable.
- `super_learner_weights*.pkl`, `table_super_learner_weights*.tex`: saved IRM prediction weights, identified by source fit and specification.

Super Learner weights are distinct from p. They summarize nuisance prediction ensembles across folds and repetitions, separately for each nuisance. Derived ATU has no independently fitted ensemble. APOS weights are not inferred from IRM weights.

LaTeX table dependencies: booktabs and adjustbox for publication tables; longtable for detailed numeric tables. The standalone PDF uses the local pdflatex installation.
