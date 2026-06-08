"""
Publication-quality LaTeX tables for the MICS DDML Analysis.

Generates tables for:
- Main DDML results (by outcome)
- Subgroup analysis (by RiskSource)
- Falsification test (RiskSource=0)
- Coefficient stability
- Leave-one-out confounders
- Stacking weights
- Sensitivity (RV/RVa)
- Mundlak correction
- Water storage + handwashing robustness
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from config import (
    OUTPUT_DIR, LEARNER_LABELS, SUBGROUP_LABELS,
    HH_OUTCOMES, U5_OUTCOMES, ANY_TREATMENT, ALL_TREATMENTS,
    N_FOLDS, N_REP,
)


def _format_coef(coef: float, se: float, ci_lower: float = None, ci_upper: float = None) -> str:
    """Format coefficient with significance stars."""
    if se == 0 or se != se:  # NaN check
        return f"{coef:.3f}"
    t = abs(coef / se)
    stars = ""
    if t > 2.576:
        stars = "***"
    elif t > 1.960:
        stars = "**"
    elif t > 1.645:
        stars = "*"
    return f"{coef:.3f}{stars}"


def _format_se(se: float) -> str:
    """Format standard error in parentheses."""
    return f"({se:.3f})"


def _format_ci(ci_lower: float, ci_upper: float) -> str:
    """Format confidence interval."""
    return f"[{ci_lower:.3f}, {ci_upper:.3f}]"


def _format_rv(rv: float) -> str:
    """Format robustness value as percentage."""
    if rv is None or rv != rv:  # NaN check
        return "---"
    return f"{rv * 100:.1f}\\%"


def _get_outcome_label(outcome_var: str) -> str:
    """Get human-readable outcome label."""
    labels = {
        "SomeRiskHome": "Some Risk Home (E.coli $>$ 0)",
        "VeryHighRiskHome": "Very High Risk Home (E.coli $\\geq$ 101)",
        "diarrhea": "Diarrhea (under-5)",
    }
    return labels.get(outcome_var, outcome_var)


def _get_treatment_label(treatment_var: str) -> str:
    """Get human-readable treatment label."""
    labels = {
        "water_treatment": "Any Treatment",
        "treat_boil": "Boil",
        "treat_chlorine": "Chlorine",
        "treat_filter": "Filter",
        "treat_other": "Other",
    }
    return labels.get(treatment_var, treatment_var)


def create_main_table(
    results: List[Dict[str, Any]],
    filename: str = "main_results.tex",
    dataset_type: str = "HH",
) -> Path:
    """
    Create main DDML results table.

    Rows: outcome × treatment
    Columns: learners (OLS, Lasso, Ridge, ENet, RF, XGB, Stacked)
    Each cell: coefficient with stars, (SE), N, RV
    """
    learner_names = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]

    # Organize results by outcome and treatment
    organized = {}
    for r in results:
        if r.get("dataset_type") != dataset_type:
            continue
        if r.get("subgroup_var") is not None:
            continue
        key = (r["outcome"], r["treatment"])
        if key not in organized:
            organized[key] = {}
        organized[key][r["learner"]] = r

    # Get unique outcomes and treatments
    outcome_vars = []
    treatment_vars = []
    for (o, t) in organized.keys():
        if o not in outcome_vars:
            outcome_vars.append(o)
        if t not in treatment_vars:
            treatment_vars.append(t)

    # Build LaTeX
    ncols = 1 + len(learner_names)
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{DDML Estimates: Effect of Water Treatment on E.coli Contamination and Diarrhea}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{ll" + "c" * len(learner_names) + "}")
    lines.append("\\hline\\hline")
    header = "Outcome & Treatment & " + " & ".join([LEARNER_LABELS.get(ln, ln) for ln in learner_names]) + " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for outcome in outcome_vars:
        for ti, treatment in enumerate(treatment_vars):
            key = (outcome, treatment)
            if key not in organized:
                continue

            olabel = _get_outcome_label(outcome) if ti == 0 else ""
            tlabel = _get_treatment_label(treatment)

            # Coefficient row
            coef_row = f"{olabel} & {tlabel}"
            for ln in learner_names:
                if ln in organized[key]:
                    r = organized[key][ln]
                    coef_row += " & " + _format_coef(r["coef"], r["se"], r.get("ci_lower"), r.get("ci_upper"))
                else:
                    coef_row += " & ---"
            coef_row += " \\\\"
            lines.append(coef_row)

            # SE row
            se_row = " & "
            for ln in learner_names:
                if ln in organized[key]:
                    r = organized[key][ln]
                    se_row += " & " + _format_se(r["se"])
                else:
                    se_row += " & "
            se_row += " \\\\"
            lines.append(se_row)

            # N row
            n_row = " & "
            for ln in learner_names:
                if ln in organized[key]:
                    r = organized[key][ln]
                    n_row += f" & {{N={r['n']:,}}}"
                else:
                    n_row += " & "
            n_row += " \\\\"
            lines.append(n_row)

            if ti < len(treatment_vars) - 1:
                lines.append("\\addlinespace")

        # Add panel divider between outcomes
        if outcome != outcome_vars[-1]:
            lines.append("\\midrule")

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} DDML IRM estimates with ATE score. ")
    lines.append("Standard errors clustered at PSU level. ")
    lines.append("$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1. ")
    lines.append(f"Cross-fitting: {N_FOLDS} folds, {N_REP} repetitions. ")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def create_subgroup_table(
    results: List[Dict[str, Any]],
    filename: str = "subgroup_results.tex",
    dataset_type: str = "HH",
) -> Path:
    """
    Create subgroup analysis table by RiskSource level.

    Three panels: No Risk, Moderate Risk, Very High Risk.
    """
    learner_names = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]
    subgroups = [(0, "No Risk"), (1, "Moderate Risk"), (2, "Very High Risk")]

    organized = {}
    for r in results:
        if r.get("dataset_type") != dataset_type:
            continue
        if r.get("subgroup_var") != "RiskSource":
            continue
        rs = r.get("subgroup_val")
        key = (rs, r["outcome"], r["treatment"])
        if key not in organized:
            organized[key] = {}
        organized[key][r["learner"]] = r

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{DDML Estimates by Source Water Risk Level}")
    lines.append("\\label{tab:subgroup_results}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{lll" + "c" * len(learner_names) + "}")
    lines.append("\\hline\\hline")
    lines.append("Risk Level & Outcome & Treatment & " + " & ".join([LEARNER_LABELS.get(ln, ln) for ln in learner_names]) + " \\\\")
    lines.append("\\hline")

    for rs_val, rs_label in subgroups:
        lines.append(f"\\multicolumn{{1}}{{l}}{{\\textit{{Panel {rs_val + 1}: {rs_label}}}}} & & & \\\\[-3pt]")
        for outcome in ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]:
            for treatment in ["water_treatment"]:
                key = (rs_val, outcome, treatment)
                if key not in organized:
                    continue

                olabel = _get_outcome_label(outcome)
                tlabel = _get_treatment_label(treatment)

                coef_row = f" & {olabel} & {tlabel}"
                for ln in learner_names:
                    if ln in organized[key]:
                        r = organized[key][ln]
                        coef_row += " & " + _format_coef(r["coef"], r["se"])
                    else:
                        coef_row += " & ---"
                coef_row += " \\\\"
                lines.append(coef_row)

                se_row = " & & "
                for ln in learner_names:
                    if ln in organized[key]:
                        r = organized[key][ln]
                        se_row += " & " + _format_se(r["se"])
                    else:
                        se_row += " & "
                se_row += " \\\\"
                lines.append(se_row)

        if rs_val < 2:
            lines.append("\\midrule")

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} DDML IRM estimates stratified by source water E.coli risk level. ")
    lines.append("Standard errors clustered at PSU level. ")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def create_falsification_table(
    results: List[Dict[str, Any]],
    filename: str = "falsification_results.tex",
    dataset_type: str = "HH",
) -> Path:
    """
    Create falsification test table (RiskSource=0 subsample).

    Expect null or near-zero effects when source water is clean.
    """
    learner_names = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]

    organized = {}
    for r in results:
        if r.get("dataset_type") != dataset_type:
            continue
        key = (r["outcome"], r["treatment"])
        if key not in organized:
            organized[key] = {}
        organized[key][r["learner"]] = r

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Falsification Test: Treatment Effects When Source Water Has No E.coli Contamination}")
    lines.append("\\label{tab:falsification}")
    lines.append("\\begin{tabular}{ll" + "c" * len(learner_names) + "}")
    lines.append("\\hline\\hline")
    lines.append("Outcome & Treatment & " + " & ".join([LEARNER_LABELS.get(ln, ln) for ln in learner_names]) + " \\\\")
    lines.append("\\hline")

    for outcome in ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]:
        for treatment in ["water_treatment"]:
            key = (outcome, treatment)
            if key not in organized:
                continue
            olabel = _get_outcome_label(outcome)
            tlabel = _get_treatment_label(treatment)

            coef_row = f"{olabel} & {tlabel}"
            for ln in learner_names:
                if ln in organized[key]:
                    r = organized[key][ln]
                    coef_row += " & " + _format_coef(r["coef"], r["se"])
                else:
                    coef_row += " & ---"
            coef_row += " \\\\"
            lines.append(coef_row)

            se_row = " & "
            for ln in learner_names:
                if ln in organized[key]:
                    r = organized[key][ln]
                    se_row += " & " + _format_se(r["se"])
                else:
                    se_row += " & "
            se_row += " \\\\"
            lines.append(se_row)

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} Falsification test. DDML estimates restricted to households with no detectable E.coli at the water source (RiskSource=0). ")
    lines.append("A well-specified model should yield near-zero effects in this subsample, as there is no contamination to remove. ")
    lines.append("Standard errors clustered at PSU level. ")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def create_stability_table(
    results: List[Dict[str, Any]],
    filename: str = "stability_results.tex",
    dataset_type: str = "HH",
) -> Path:
    """
    Create coefficient stability table showing how ATE changes
    as confounder groups are progressively added.
    """
    organized = {}
    for r in results:
        key = (r["outcome"], r["treatment"], r.get("stability_step", "unknown"))
        organized[key] = r

    steps = [s[0] for s in STABILITY_GROUPS]
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Coefficient Stability: Progressive Addition of Confounders}")
    lines.append("\\label{tab:stability}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\hline\\hline")
    lines.append("Confounder Set & Outcome & Coefficient & SE \\\\")
    lines.append("\\hline")

    for outcome in ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]:
        for treatment in ["water_treatment"]:
            for step in steps:
                key = (outcome, treatment, step)
                if key in organized:
                    r = organized[key]
                    olabel = _get_outcome_label(outcome)
                    lines.append(f"{step} & {olabel} & {_format_coef(r['coef'], r['se'])} & {_format_se(r['se'])} \\\\")
            lines.append("\\addlinespace")

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} Stacked ensemble DDML estimates. ")
    lines.append("Coefficients show how the estimated ATE changes as confounder groups are progressively added. ")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def create_loo_table(
    results: List[Dict[str, Any]],
    filename: str = "loo_results.tex",
    dataset_type: str = "HH",
) -> Path:
    """
    Create leave-one-out confounder table.

    Each row shows the ATE when dropping one confounder group from the full set.
    """
    organized = {}
    for r in results:
        key = (r["outcome"], r["treatment"], r.get("loo_dropped", "unknown"))
        organized[key] = r

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Leave-One-Out Confounder Robustness}")
    lines.append("\\label{tab:loo}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\hline\\hline")
    lines.append("Dropped Group & Outcome & Coefficient & SE \\\\")
    lines.append("\\hline")

    for outcome in ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]:
        for treatment in ["water_treatment"]:
            key_full = (outcome, treatment, "Full")
            if key_full in organized:
                r = organized[key_full]
                lines.append(f"Full specification & {_get_outcome_label(outcome)} & {_format_coef(r['coef'], r['se'])} & {_format_se(r['se'])} \\\\")
            for dropped in ["wealth", "education", "urban", "sanitation", "water_source", "country", "source_ecoli", "child"]:
                key = (outcome, treatment, dropped)
                if key in organized:
                    r = organized[key]
                    lines.append(f"  $-$ {dropped.replace('_', ' ').title()} & & {_format_coef(r['coef'], r['se'])} & {_format_se(r['se'])} \\\\")
        lines.append("\\addlinespace")

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def export_all_results_csv(
    all_results: Dict[str, List[Dict[str, Any]]],
    filename: str = "all_results.csv",
) -> Path:
    """Export all results to a single CSV file (excluding model objects)."""
    rows = []
    for analysis_name, results in all_results.items():
        for r in results:
            row = {k: v for k, v in r.items() if k != "model"}
            row["analysis"] = analysis_name
            rows.append(row)

    df = pd.DataFrame(rows)
    filepath = OUTPUT_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"All results CSV saved to: {filepath}")
    return filepath


def create_apos_table(
    results: List[Dict[str, Any]],
    filename: str = "table_apos.tex",
) -> Path:
    """APOS contrasts: each specific treatment vs. no treatment, all outcomes.

    Expects result dicts from ``apos.run_apos`` (keys: outcome, treatment,
    coef, se, ci_lower, ci_upper, n_treated).
    """
    outcomes = []
    for r in results:
        if r["outcome"] not in outcomes:
            outcomes.append(r["outcome"])

    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Specific Treatment Effects vs. No Treatment (APOS, Stacked Ensemble)}",
             "\\label{tab:apos}",
             "\\begin{tabular}{llccc}", "\\hline\\hline",
             "Outcome & Treatment & ATE & 95\\% CI & N (treated) \\\\", "\\hline"]
    for o in outcomes:
        rows = [r for r in results if r["outcome"] == o]
        for i, r in enumerate(rows):
            olab = _get_outcome_label(o) if i == 0 else ""
            lines.append(
                f"{olab} & {r['treatment']} & {_format_coef(r['coef'], r['se'])} "
                f"& {_format_ci(r['ci_lower'], r['ci_upper'])} & {r['n_treated']:,} \\\\"
            )
            lines.append(f" & & {_format_se(r['se'])} & & \\\\")
        if o != outcomes[-1]:
            lines.append("\\addlinespace")
    lines += ["\\hline\\hline", "\\end{tabular}",
              "\\begin{minipage}{\\textwidth}\\footnotesize",
              "\\textit{Notes:} Average Potential Outcome (AIPW) contrasts, "
              "ATE($d$) $=$ E[$Y(d)$]$-$E[$Y(0)$], estimated jointly on the full sample. "
              "Stacked ensemble for both nuisances; multi-class propensity P($D{=}d\\mid X$). "
              "SEs from \\texttt{causal\\_contrast} (i.i.d.; DoubleMLAPOS does not support clustering). "
              "$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.",
              "\\end{minipage}", "\\end{table}"]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


def create_sensitivity_table(
    results: List[Dict[str, Any]],
    filename: str = "table_sensitivity.tex",
    learner: str = "stacked",
) -> Path:
    """Sensitivity to unobserved confounding (RV / RVa) for the IRM headline learner.

    Consumes the same result dicts as ``create_main_table`` (rv_q / rv_qa are
    populated by ``models.estimate_effect``).
    """
    rows = [
        r for r in results
        if r.get("learner") == learner
        and r.get("subgroup_var") is None
        and r.get("rv_q") is not None
    ]

    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Sensitivity to Unobserved Confounding (Stacked IRM)}",
             "\\label{tab:sensitivity}",
             "\\begin{tabular}{llccc}", "\\hline\\hline",
             "Outcome & Treatment & ATE & RV & RV$_\\alpha$ \\\\", "\\hline"]
    for r in rows:
        lines.append(
            f"{_get_outcome_label(r['outcome'])} & {_get_treatment_label(r['treatment'])} "
            f"& {_format_coef(r['coef'], r['se'])} "
            f"& {_format_rv(r['rv_q'])} & {_format_rv(r['rv_qa'])} \\\\"
        )
    lines += ["\\hline\\hline", "\\end{tabular}",
              "\\begin{minipage}{\\textwidth}\\footnotesize",
              "\\textit{Notes:} Chernozhukov et al. (2022) sensitivity, worst case $\\rho=1$. "
              "RV is the share of residual variance in both treatment and outcome an unobserved "
              "confounder would need to drive the point estimate to zero; RV$_\\alpha$ to bring the "
              "95\\% CI to zero.",
              "\\end{minipage}", "\\end{table}"]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


import pandas as pd
from config import STABILITY_GROUPS