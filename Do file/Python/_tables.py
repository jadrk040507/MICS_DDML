"""LaTeX tables for the active stacked water-treatment analysis."""

from pathlib import Path
import gc
import pandas as pd
import numpy as np
from _functions import relative_robustness_value


def _format_coef(coef: float, se: float) -> str:
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


def _latex_text(value) -> str:
    """Escape plain-text underscores before inserting text into LaTeX."""
    return str(value).replace("_", r"\_")


def _format_rv(rv: float) -> str:
    """Format robustness value as percentage."""
    if rv is None or rv != rv:  # NaN check
        return "---"
    return f"{rv * 100:.4f}\\%"


def _get_outcome_label(outcome_var: str) -> str:
    """Get human-readable outcome label."""
    labels = {
        "SomeRiskHome": r"Some Risk Home (E.coli $>0$ CFU)",
        "VeryHighRiskHome": r"Very High Risk Home (E.coli $>100$ CFU)",
        "diarrhea": "Diarrhea (under-5)",
    }
    return labels.get(outcome_var, outcome_var)


_RELATIVE_SENSITIVITY_TREATMENTS = {
    "Any Treatment": "Any Treatment",
    "Boiling": "Boiling",
    "Chlorination/tablets": "Chlorination/tablets",
    "Straining/settling": "Straining/settling",
}

_SHARE_LABELS = {
    0: "No treatment",
    1: "Boiling",
    2: "Aquatabs",
    3: "Straining/settling",
    98: "Other treatment",
}


def _relative_sensitivity_treatment(value):
    """Return the canonical treatment label when it is part of the table."""

    return _RELATIVE_SENSITIVITY_TREATMENTS.get(str(value).strip())


def _relative_sensitivity_value(row, reduced_key, full_key, relative=False):
    """Return either a reduced RV or its ratio to the full RV.

    The full-specification column is handled by the table builder. For a
    dropped-control column, absolute rows use the reduced value and relative
    rows use reduced divided by full. Keeping this distinction here prevents
    the two specifications from being accidentally displayed as identical.
    """

    reduced = row.get(reduced_key)
    full = row.get(full_key)
    if relative:
        return relative_robustness_value(reduced, full)
    return reduced


def create_relative_sensitivity_tables(
    results,
    filename_prefix="table_sensitivity_relative",
    learner="stacked",
    method=None,
    output_dir=None,
    group_order=None,
):
    """Write panelled RV/RV-alpha tables relative to the full specification.

    The input contains one row per outcome/treatment/omitted-control-group.
    ``rv_q`` and ``rv_qa`` are the full-specification values; the reduced
    specification values must be supplied as ``rv_q_without`` and
    ``rv_qa_without``.  The first column is the complete specification, so its
    relative values are always 1.00 and its absolute values are the original
    RV/RV-alpha.  ``Other`` treatment methods are omitted.

    One file is written per outcome because columns are omitted control groups
    and the U5 outcome has the additional child-age/sex group.
    """

    import pandas as pd

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if learner is not None and "learner" in frame:
        frame = frame[frame["learner"].eq(learner)]
    if method is not None and "method" in frame:
        frame = frame[frame["method"].eq(method)]
    if frame.empty:
        raise ValueError("No sensitivity rows remain after learner/method filters.")

    frame = frame.copy()
    frame["_treatment_label"] = frame["treatment"].map(
        _relative_sensitivity_treatment
    )
    frame = frame[frame["_treatment_label"].notna()]
    if frame.empty:
        raise ValueError("No supported treatment rows found for relative sensitivity tables.")

    if group_order is None:
        group_order = [
            "wealth", "country", "urban", "water_source", "hh_demog",
            "sanitation", "source_ecoli", "child_age_sex",
        ]
    group_rank = {key: i for i, key in enumerate(group_order)}
    outcomes = list(dict.fromkeys(frame["outcome"].tolist()))
    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for outcome in outcomes:
        sub = frame[frame["outcome"].eq(outcome)].copy()
        groups = list(dict.fromkeys(sub["group"].tolist()))
        groups.sort(key=lambda key: (group_rank.get(key, len(group_rank)), str(key)))
        group_labels = {
            row["group"]: row.get("group_label", row["group"])
            for _, row in sub.iterrows()
        }
        treatments = [
            label for label in [
                "Any Treatment", "Boiling", "Chlorination/tablets",
                "Straining/settling",
            ] if label in set(sub["_treatment_label"])
        ]

        ncol = 1 + len(groups)

        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}",
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Sensitivity relative to the complete specification: {_get_outcome_label(outcome)}}}",
            rf"\label{{tab:{filename_prefix}-{outcome}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{{'l' + 'c' * ncol}}}",
            r"\toprule",
            " & ".join(["", "Full specification"] + [group_labels[g] for g in groups]) + r" \\",
            r"\midrule",
        ]

        for treatment_index, treatment_label in enumerate(treatments):
            tsub = sub[sub["_treatment_label"].eq(treatment_label)]
            full = tsub.iloc[0]
            lines.append(rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{{treatment_label}}}}} \\")

            rows = [
                (r"RV original", "rv_q", "rv_q_without", False),
                (r"RV relative", "rv_q", "rv_q_without", True),
                (r"RV$_\alpha$ original", "rv_qa", "rv_qa_without", False),
                (r"RV$_\alpha$ relative", "rv_qa", "rv_qa_without", True),
            ]
            for label, full_key, reduced_key, relative in rows:
                values = ["1.0000" if relative else _format_rv(full.get(full_key))]
                for group in groups:
                    hit = tsub[tsub["group"].eq(group)]
                    if hit.empty:
                        values.append("---")
                        continue
                    row = hit.iloc[0]
                    value = _relative_sensitivity_value(
                        row, reduced_key, full_key, relative=relative
                    )
                    values.append(
                        "---" if value is None or pd.isna(value)
                        else (f"{value:.4f}" if relative else _format_rv(value))
                    )
                lines.append(" & ".join([rf"\quad {label}"] + values) + r" \\")

            if treatment_index < len(treatments) - 1:
                lines.append(r"\midrule")

        notes = (
            r"\textit{Notes:} Relative RV values equal the RV from the specification "
            r"without the indicated control group divided by the RV from the complete "
            r"specification. Absolute RV and RV$_\alpha$ values are shown as percentages. "
            r"A relative value below one indicates lower robustness after removing the "
            r"group; ratios are suppressed when the complete-specification denominator is "
            r"effectively zero. Other treatment methods are omitted."
        )
        lines += [
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}", r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{outcome}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")

    return paths


def create_combined_sensitivity_tables(
    results,
    filename_prefix="table_sensitivity",
    learner="stacked",
    output_dir=None,
):
    """Write one clustered sensitivity table per outcome.

    Each table has Panel A for IRM and Panel B for APOS. Absolute and relative
    RV/RV-alpha rows are shown together so readers can see both the robustness
    level and the change caused by omitting the control group.
    """

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if learner is not None and "learner" in frame:
        frame = frame[frame["learner"].eq(learner)]
    frame = frame[frame["specification"].eq("clustered_folds")].copy()
    frame["_treatment_label"] = frame["treatment"].map(
        _relative_sensitivity_treatment
    )
    frame = frame[frame["_treatment_label"].notna()]
    if frame.empty:
        raise ValueError("No clustered sensitivity rows remain.")

    group_order = [
        "wealth", "country", "urban", "water_source", "hh_demog",
        "sanitation", "source_ecoli", "child_age_sex",
    ]
    rank = {key: i for i, key in enumerate(group_order)}
    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for outcome in dict.fromkeys(frame["outcome"].tolist()):
        sub = frame[frame["outcome"].eq(outcome)]
        groups = sorted(
            sub["group"].unique(),
            key=lambda key: (rank.get(key, len(rank)), str(key)),
        )
        labels = {
            row["group"]: row.get("group_label", row["group"])
            for _, row in sub.iterrows()
        }
        ncol = 1 + len(groups)
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}", r"\begin{table}[p]", r"\centering",
            rf"\caption{{Sensitivity relative to the complete specification: {_get_outcome_label(outcome)}}}",
            rf"\label{{tab:{filename_prefix}-{outcome}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{{'l' + 'c' * ncol}}}", r"\toprule",
            " & ".join(["", "Full specification"] + [labels[g] for g in groups]) + r" \\",
            r"\midrule",
        ]
        rows = [
            (r"RV original", "rv_q", "rv_q_without", False),
            (r"RV relative", "rv_q", "rv_q_without", True),
            (r"RV$_\alpha$ original", "rv_qa", "rv_qa_without", False),
            (r"RV$_\alpha$ relative", "rv_qa", "rv_qa_without", True),
        ]
        panels = [
            ("IRM", "Any Treatment"),
            ("APOS", None),
        ]
        for panel_index, (method, fixed_treatment) in enumerate(panels):
            method_sub = sub[sub["method"].eq(method)]
            lines.append(
                rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{Panel {'AB'[panel_index]}: {method}}}}} \\"
            )
            treatments = [fixed_treatment] if fixed_treatment else [
                label for label in [
                    "Boiling", "Chlorination/tablets", "Straining/settling",
                ] if label in set(method_sub["_treatment_label"])
            ]
            for treatment_index, treatment in enumerate(treatments):
                tsub = method_sub[method_sub["_treatment_label"].eq(treatment)]
                if tsub.empty:
                    continue
                full = tsub.iloc[0]
                lines.append(
                    rf"\multicolumn{{{ncol + 1}}}{{l}}{{\quad\textit{{{treatment}}}}} \\"
                )
                for label, full_key, reduced_key, relative in rows:
                    values = ["1.0000" if relative else _format_rv(full.get(full_key))]
                    for group in groups:
                        hit = tsub[tsub["group"].eq(group)]
                        value = None if hit.empty else _relative_sensitivity_value(
                            hit.iloc[0], reduced_key, full_key, relative=relative
                        )
                        values.append(
                            "---" if value is None or pd.isna(value)
                            else (f"{value:.4f}" if relative else _format_rv(value))
                        )
                    lines.append(" & ".join([rf"\quad {label}"] + values) + r" \\")
                if treatment_index < len(treatments) - 1:
                    lines.append(r"\addlinespace")
            if panel_index < len(panels) - 1:
                lines.append(r"\midrule")

        notes = (
            r"\textit{Notes:} Panel A reports IRM and Panel B reports APOS. "
            r"Relative values equal the leave-one-control-group-out RV or "
            r"RV$_\alpha$ divided by the complete-specification value. "
            r"Absolute values are percentages; all values use four decimals. "
            r"Estimation uses cluster-level folds."
        )
        lines += [
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}", r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{outcome}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")
    return paths


def create_benchmark_sensitivity_tables(results, filename_prefix="table_sensitivity_benchmark", output_dir=None):
    """Write source-E.coli benchmark tables in the appendix table style.

    The table is deliberately organized in IRM and APOS panels, with fold
    specifications shown as sub-blocks.  This keeps the benchmark statistics
    together without putting specification and treatment names into a wide,
    difficult-to-scan header.
    """

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    treatment_labels = {
        "0": "No treatment", "1": "Boiling", "2": "Aquatabs",
        "3": "Straining/settling", "98": "Other treatment",
    }

    def treatment_label(value, method):
        text = str(value).strip()
        code = text.split()[0] if text else text
        if method == "IRM":
            return "Water treatment"
        return treatment_labels.get(code, text)

    specification_labels = {
        "clustered_folds": "Clustered folds",
        "unclustered": "Ordinary folds",
    }
    metric_headers = [
        r"RV", r"RV$_\alpha$", r"\shortstack{Omit source E.coli\\$cf_y$}",
        r"\shortstack{Omit source E.coli\\$cf_d$}", r"$\rho$",
        r"\shortstack{Omit source E.coli\\$\Delta\theta$}",
    ]

    for outcome in frame["outcome"].drop_duplicates():
        sub = frame[frame["outcome"].eq(outcome)]
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}", r"\begin{table}[p]", r"\centering",
            rf"\caption{{Sensitivity to omission of source E.coli deciles: {_get_outcome_label(outcome)}}}",
            rf"\label{{tab:{filename_prefix}-{outcome}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            "Panel & Specification & Treatment & " + " & ".join(metric_headers) + r" \\",
            r"\midrule",
        ]

        for panel_index, method in enumerate(("IRM", "APOS")):
            method_sub = sub[sub["method"].eq(method)]
            if method_sub.empty:
                continue
            panel_label = "Panel A: IRM" if method == "IRM" else "Panel B: APOS"
            lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{panel_label}}}}} \\")

            for spec_index, specification in enumerate(("clustered_folds", "unclustered")):
                spec_sub = method_sub[method_sub["specification"].eq(specification)]
                if spec_sub.empty:
                    continue
                for _, row in spec_sub.iterrows():
                    treatment = treatment_label(row["treatment"], method)
                    values = [
                        f"{100 * row['rv']:.4f}\\%",
                        f"{100 * row['rva']:.4f}\\%",
                        f"{row['cf_y']:.4f}", f"{row['cf_d']:.4f}",
                        f"{row['rho']:.4f}", f"{row['delta_theta']:.4f}",
                    ]
                    lines.append(
                        f" & {_latex_text(specification_labels[specification])} & "
                        f"{_latex_text(treatment)} & " + " & ".join(values) + r" \\")
                if spec_index == 0 and not method_sub[method_sub["specification"].eq("unclustered")].empty:
                    lines.append(r"\addlinespace[3pt]")
            if panel_index == 0 and not sub[sub["method"].eq("APOS")].empty:
                lines.append(r"\midrule")

        notes = (
            r"\textit{Notes:} The omitted controls are the initial source-water "
            r"E.coli decile indicators ($source\_ecoli\_*$). Panel A reports the "
            r"IRM estimate for water treatment; Panel B reports APOS contrasts for "
            r"Boiling, Aquatabs, Straining/settling, and Other treatment. Clustered "
            r"folds use cluster-level sample splitting; ordinary folds use unclustered "
            r"folds. $cf_y$ and $cf_d$ measure the observed gains in outcome and "
            r"treatment prediction, respectively. $\rho$ is the implied signed adversity "
            r"parameter, and $\Delta\theta = \theta_{\mathrm{omit}} - \theta_{\mathrm{full}}$. "
            r"RV and RV$_\alpha$ are the confounding strengths needed to eliminate the "
            r"estimate or its confidence interval."
        )
        lines += [
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\scriptsize {notes}\end{{minipage}}",
            r"\end{table}", r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{outcome}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")
    return paths


def write_sensitivity_summary_table(results, output_dir, filename="table_sensitivity_summary.tex"):
    """Write a compact RV/RV-alpha table for the active sensitivity stage."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{DoubleML sensitivity analysis}",
        r"\label{tab:sensitivity-summary}",
        r"\begin{tabular}{llllrr}", r"\toprule",
        r"Outcome & Specification & Method & Treatment & RV & RV$_\alpha$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_latex_text(row['outcome'])} & "
            f"{_latex_text(row['specification'])} & "
            f"{_latex_text(row['method'])} & "
            f"{_latex_text(row['treatment'])} & "
            f"{100 * float(row['rv']):.4f}\\% & "
            f"{100 * float(row['rva']):.4f}\\% " + r"\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{0.9\linewidth}\footnotesize "
        r"RV is the confounding strength needed to remove the estimate; "
        r"RV$_\alpha$ is the strength needed to remove its confidence interval. "
        r"The analysis uses $cf_y = cf_d = 0.03$ and $\rho = 1$.\end{minipage}",
        r"\end{table}",
    ]
    path = Path(output_dir) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def create_relative_sensitivity_comparison_tables(
    results,
    filename_prefix="table_sensitivity_relative_appendix",
    learner="stacked",
    method=None,
    output_dir=None,
    specifications=("clustered_folds", "unclustered"),
    group_order=None,
):
    """Write two-panel relative-sensitivity tables for the appendix."""

    import pandas as pd

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if learner is not None and "learner" in frame:
        frame = frame[frame["learner"].eq(learner)]
    if method is not None and "method" in frame:
        frame = frame[frame["method"].eq(method)]
    frame = frame[frame["specification"].isin(specifications)].copy()
    frame["_treatment_label"] = frame["treatment"].map(
        _relative_sensitivity_treatment
    )
    frame = frame[frame["_treatment_label"].notna()]
    if frame.empty:
        raise ValueError("No sensitivity rows remain for the comparison table.")

    if group_order is None:
        group_order = [
            "wealth", "country", "urban", "water_source", "hh_demog",
            "sanitation", "source_ecoli", "child_age_sex",
        ]
    group_rank = {key: i for i, key in enumerate(group_order)}
    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for outcome in list(dict.fromkeys(frame["outcome"].tolist())):
        sub = frame[frame["outcome"].eq(outcome)]
        groups = list(dict.fromkeys(sub["group"].tolist()))
        groups.sort(key=lambda key: (group_rank.get(key, len(group_rank)), str(key)))
        group_labels = {
            row["group"]: row.get("group_label", row["group"])
            for _, row in sub.iterrows()
        }
        ncol = 1 + len(groups)
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}", r"\begin{table}[p]", r"\centering",
            rf"\caption{{Sensitivity relative to the complete specification: {_get_outcome_label(outcome)}}}",
            rf"\label{{tab:{filename_prefix}-{outcome}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{{'l' + 'c' * ncol}}}", r"\toprule",
            " & ".join(["", "Full specification"] + [group_labels[g] for g in groups]) + r" \\",
            r"\midrule",
        ]

        panel_labels = {
            "clustered_folds": "Panel A: Cluster-level folds",
            "unclustered": "Panel B: Ordinary folds",
        }
        for panel_index, specification in enumerate(specifications):
            spec_sub = sub[sub["specification"].eq(specification)]
            if spec_sub.empty:
                continue
            lines.append(
                rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{{panel_labels.get(specification, specification)}}}}} \\")
            treatments = [
                label for label in [
                    "Any Treatment", "Boiling", "Chlorination/tablets",
                    "Straining/settling",
                ] if label in set(spec_sub["_treatment_label"])
            ]
            for treatment_index, treatment_label in enumerate(treatments):
                tsub = spec_sub[spec_sub["_treatment_label"].eq(treatment_label)]
                full = tsub.iloc[0]
                lines.append(
                    rf"\multicolumn{{{ncol + 1}}}{{l}}{{\quad\textit{{{treatment_label}}}}} \\")
                for label, full_key, reduced_key, relative in [
                    (r"RV original", "rv_q", "rv_q_without", False),
                    (r"RV relative", "rv_q", "rv_q_without", True),
                    (r"RV$_\alpha$ original", "rv_qa", "rv_qa_without", False),
                    (r"RV$_\alpha$ relative", "rv_qa", "rv_qa_without", True),
                ]:
                    values = ["1.0000" if relative else _format_rv(full.get(full_key))]
                    for group in groups:
                        hit = tsub[tsub["group"].eq(group)]
                        value = None if hit.empty else _relative_sensitivity_value(
                            hit.iloc[0], reduced_key, full_key, relative=relative
                        )
                        values.append(
                            "---" if value is None or pd.isna(value)
                            else (f"{value:.4f}" if relative else _format_rv(value))
                        )
                    lines.append(" & ".join([rf"\quad {label}"] + values) + r" \\")
                if treatment_index < len(treatments) - 1:
                    lines.append(r"\addlinespace")
            if panel_index < len(specifications) - 1:
                lines.append(r"\midrule")

        notes = (
            r"\textit{Notes:} Relative values are the leave-one-control-group-out "
            r"RV or RV$_\alpha$ divided by the complete-specification value. "
            r"Absolute values are percentages. Panel A uses cluster-level folds; "
            r"Panel B uses ordinary folds. APOS Panel A SEs use the cluster sandwich. "
            r"Other treatment methods are omitted."
        )
        lines += [
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}", r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{outcome}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")

    return paths


def _write_column_publication_table(
    output_dir,
    estimates,
    outcome_order,
    filename,
    caption,
    label,
    report_levels,
    folds,
    repetitions,
    specifications=("clustered",),
) -> Path:
    """Write publication results with outcomes arranged in columns."""

    model_suffix = {"clustered": "cluster", "unclustered": "no_cluster"}
    specification_labels = {
        "clustered": "Clustered folds",
        "unclustered": "Ordinary folds",
    }
    short_outcome_labels = {
        "SomeRiskHome": "Some risk",
        "VeryHighRiskHome": "Very high risk",
        "diarrhea": "Diarrhea (U5)",
    }
    short_specification_labels = {
        "clustered": "Clustered",
        "unclustered": "Unclustered",
    }

    def result_cell(summary, row_number=0, se_override=None):
        row = summary.iloc[row_number]
        se = float(row["std err"]) if se_override is None else float(se_override)
        return _format_coef(float(row["coef"]), se), _format_se(se)

    def sample_statistics(frame, outcome, treatment):
        shares = frame[treatment].value_counts(normalize=True).reindex(
            [0, 1, 2, 3, 98], fill_value=0.0
        )
        no_treatment = frame[treatment].eq(0)
        return {
            "n": len(frame),
            "psu": frame["Cluster_var"].nunique() if "Cluster_var" in frame else None,
            # Use the same no-treatment reference group as the causal contrast.
            "mean_y": float(frame.loc[no_treatment, outcome].mean()),
            "mean_d": float(frame[treatment].mean()) if treatment == "water_treatment" else None,
            "shares": shares,
        }

    def value_text(value):
        return "---" if value is None else f"{value:,}"

    columns = []
    for outcome in outcome_order:
        dataset_name = "U5" if outcome == "diarrhea" else "HH"
        models = estimates[(dataset_name, outcome)]
        for specification in specifications:
            suffix = model_suffix[specification]
            irm = models[f"irm_{suffix}"]
            apos = models[f"apos_{suffix}"].causal_contrast(reference_levels=[0])
            apos_summary = apos.summary.iloc[:len(report_levels) - 1]
            apos_se = models.get("apos_cluster_se") if specification == "clustered" else None
            columns.append({
                "outcome": outcome,
                "outcome_label": _get_outcome_label(outcome),
                "specification": specification,
                "irm": irm.summary,
                "apos": apos_summary,
                "apos_se": apos_se,
                "irm_stats": sample_statistics(models[f"irm_frame_{'cluster' if specification == 'clustered' else 'no_cluster'}"], outcome, "water_treatment"),
                "apos_stats": sample_statistics(models[f"apos_frame_{'cluster' if specification == 'clustered' else 'no_cluster'}"], outcome, "WQ15_g"),
            })
            del irm, apos
            gc.collect()
        del models
        gc.collect()

    n_columns = len(columns)
    lines = [
        r"% Requires: \usepackage{booktabs, graphicx, adjustbox}",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{l{'c' * n_columns}}}",
        r"\hline\hline",
    ]

    if len(specifications) == 1:
        lines.append(" & " + " & ".join(short_outcome_labels[c["outcome"]] for c in columns) + r" \\")
    else:
        header = [""]
        for outcome in outcome_order:
            header.append(rf"\multicolumn{{{len(specifications)}}}{{c}}{{{short_outcome_labels[outcome]}}}")
        lines.append(" & ".join(header) + r" \\")
        lines.append(" & " + " & ".join(short_specification_labels[c["specification"]] for c in columns) + r" \\")
    lines += [r"\hline", r"\multicolumn{" + str(1 + n_columns) + r"}{l}{\textit{IRM}} \\,"]

    # The IRM block marker must end with a LaTeX row break, not punctuation.
    lines[-1] = lines[-1].replace(",", "")

    def row(label, values):
        return label + " & " + " & ".join(values) + " " + chr(92) * 2

    def once_per_outcome(values):
        """Show sample/descriptive values once across appendix subcolumns."""

        if len(specifications) == 1:
            return values
        output = []
        seen = set()
        for column, value in zip(columns, values):
            outcome = column["outcome"]
            output.append(value if outcome not in seen else "")
            seen.add(outcome)
        return output

    lines.append(row(r"Any treatment", [result_cell(c["irm"])[0] for c in columns]))
    lines.append(row("", [result_cell(c["irm"])[1] for c in columns]))
    lines.append(r"\addlinespace[6pt]")
    lines.append(r"\multicolumn{" + str(1 + n_columns) + r"}{l}{\textit{Shares}} \\")
    # IRM is a binary-treatment model, so this is the outcome mean, not a
    # treatment share. Treatment prevalence is reported separately below.
    lines.append(row(r"Y mean (no treatment) (%)", once_per_outcome([f"{100 * c['irm_stats']['mean_y']:.1f}\\%" for c in columns])))
    lines.append(row(r"Treated (\%)", once_per_outcome([f"{100 * c['irm_stats']['mean_d']:.1f}\\%" for c in columns])))

    lines += [r"\midrule", r"\multicolumn{" + str(1 + n_columns) + r"}{l}{\textit{APOS}} \\"]
    treatment_labels = {
        0: "Boiling",
        1: "Chlorination/tablets",
        2: "Straining/settling",
    }
    for index, contrast in enumerate(columns[0]["apos"].index):
        values = []
        ses = []
        for c in columns:
            se_override = c["apos_se"][index] if c["apos_se"] is not None else None
            coef, se = result_cell(c["apos"], index, se_override)
            values.append(coef)
            ses.append(se)
        lines.append(row(treatment_labels.get(index, str(contrast)), values))
        lines.append(row("", ses))
    lines.append(r"\addlinespace[6pt]")
    # APOS rows below are the empirical distribution of the categorical
    # treatment variable WQ15_g. Do not label those treatment shares as an
    # outcome share; the outcome mean is already reported in the IRM block.
    for level in [0, 1, 2, 3, 98]:
        lines.append(row(_SHARE_LABELS[level], once_per_outcome([f"{100 * c['apos_stats']['shares'].loc[level]:.1f}\\%" for c in columns])))
    lines += [r"\midrule", r"\multicolumn{" + str(1 + n_columns) + r"}{l}{\textit{Sample}} \\"]
    lines.append(row(r"Observations", once_per_outcome([f"{c['irm_stats']['n']:,}" for c in columns])))
    lines.append(row(r"PSUs", once_per_outcome([value_text(c["irm_stats"]["psu"]) for c in columns])))
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        rf"\begin{{minipage}}{{\linewidth}}\scriptsize \textit{{Notes:}} Cells report coefficients with significance stars and standard errors in parentheses. IRM and APOS are separated into blocks. Clustered specifications use cluster-level sample splitting; ordinary specifications use unclustered folds. Shares use readable treatment names, including Other treatment. Cross-fitting uses "
        rf"{folds} folds and {repetitions} repetitions. The outcome mean is calculated among observations with no water treatment, the counterfactual reference group. APOS effects are contrasts relative to treatment level 0. $^{{***}}p<0.01$, $^{{**}}p<0.05$, $^{{*}}p<0.1$.\end{{minipage}}",
        r"\end{table}",
    ]
    path = Path(output_dir) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_super_learner_weights_table(output_dir, estimates, outcome_order, filename):
    """Write average fitted Super Learner weights by outcome and fold type."""

    def collect_weights(model, nuisance):
        # Read the compact weights saved by mics_readable.py when the fitted
        # base learners have been removed from the checkpoint.
        compact = getattr(model, "convex_weights", {})
        if compact:
            compact_weights = compact.get(nuisance, {})
            if compact_weights:
                return {name: float(value) for name, value in compact_weights.items()}

        models = model.models or {}
        # DoubleML IRM/APOS store the outcome learner separately by treatment
        # arm (ml_g0, ml_g1, ...), while the public learner is named ml_g.
        # Pool those arm-specific fitted ensembles for the reported average.
        if nuisance == "ml_g":
            fitted = {
                key: value for key, value in models.items()
                if key.startswith("ml_g")
            }
        else:
            fitted = models.get(nuisance, {})
        weights = []

        def visit(value):
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, np.ndarray):
                # DoubleML may store fitted nuisance learners inside object
                # arrays. Traverse those arrays so stored ensemble weights are
                # found regardless of the container used by the installed
                # DoubleML version.
                for item in value.flat:
                    visit(item)
            elif hasattr(value, "weights_"):
                # ``estimators`` is the configured (name, estimator) list;
                # ``weights_`` has the corresponding fitted convex weights.
                names = [name for name, _ in value.estimators]
                weights.append(dict(zip(names, np.asarray(value.weights_))))

        visit(fitted)
        if not weights:
            raise ValueError(
                f"No stored weights found for nuisance learner {nuisance}. "
                "Refit with store_models=True."
            )
        # A learner can be absent in a particular stored fit in edge cases.
        # Average each learner over the fits in which it is present instead of
        # failing with a misleading missing-key error.
        names = list(dict.fromkeys(name for row in weights for name in row))
        return {
            name: float(np.mean([row[name] for row in weights if name in row]))
            for name in names
        }

    columns = []
    for outcome in outcome_order:
        dataset = "U5" if outcome == "diarrhea" else "HH"
        for specification, suffix in [("Clustered", "cluster"), ("Unclustered", "no_cluster")]:
            models = estimates[(dataset, outcome)]
            fitted = models[f"irm_{suffix}"]
            columns.append((outcome, specification, fitted))

    # Outcome regression and treatment propensity use different learner
    # libraries. In particular, ``g(X)`` has OLS while ``m(X)`` has logistic
    # regression. Reusing one list for both nuisances makes a valid table fail
    # when it searches for OLS among propensity-model weights.
    learner_names_by_nuisance = {
        "ml_g": ["ols", "lasso", "elastic_net", "random_forest", "xgboost"],
        "ml_m": ["logit", "lasso", "elastic_net", "random_forest", "xgboost"],
    }
    # Short headers keep the table readable in landscape without shrinking the
    # numerical cells excessively. Full outcome definitions are documented in
    # the note below the table.
    short_headers = [
        r"\shortstack{SomeRiskHome\\Clustered}",
        r"\shortstack{SomeRiskHome\\Unclustered}",
        r"\shortstack{VeryHighRiskHome\\Clustered}",
        r"\shortstack{VeryHighRiskHome\\Unclustered}",
        r"\shortstack{Diarrhea\\Clustered}",
        r"\shortstack{Diarrhea\\Unclustered}",
    ]
    lines = [
        r"% Requires: \usepackage{booktabs, graphicx, pdflscape}",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Super Learner weights}",
        r"\label{tab:super-learner-weights}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Learner & " + " & ".join(short_headers) + r" \\",
        r"\midrule",
    ]
    for panel_index, (nuisance, label) in enumerate(
        [("ml_g", "Panel A: Outcome learner $g(X)$"),
         ("ml_m", "Panel B: Treatment learner $m(X)$")]
    ):
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\")
        for learner in learner_names_by_nuisance[nuisance]:
            values = []
            for outcome, specification, model in columns:
                weights = collect_weights(model, nuisance)
                if learner not in weights:
                    # Keep the table readable if a future fit omits one
                    # learner, while making the missing cell explicit.
                    values.append("---")
                else:
                    values.append(f"{weights[learner]:.3f}")
            lines.append(
                f"{_latex_text(learner)} & "
                + " & ".join(values)
                + r" \\"
            )
        if panel_index == 0:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{0.95\linewidth}\footnotesize Weights are averaged across outer folds and repetitions. C = clustered-fold specification; U = unclustered-fold specification. SomeRiskHome denotes any detectable E. coli at home, VeryHighRiskHome denotes E. coli above 100 CFU/100 mL at home, and Diarrhea denotes diarrhea among children under five. The outcome learner uses OLS, Lasso, Elastic Net, random forest, and XGBoost; the treatment learner uses logistic regression, Lasso, Elastic Net, random forest, and XGBoost.\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ]
    path = Path(output_dir) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_super_learner_weights_tables(
    output_dir, estimates, outcome_order, weights=None
):
    """Write separate, vertically stacked weight tables by fold specification.

    Each output file contains three outcome panels. Within each panel, the
    outcome learner and treatment learner are shown side by side, which keeps
    clustered and unclustered results from competing for space in one wide
    table.
    """

    def collect_weights(model, nuisance, dataset=None, outcome=None, specification=None):
        # Read the compact results table when checkpoints have intentionally
        # discarded fitted nuisance learners.
        if weights is not None and not weights.empty:
            model_name = (
                f"{dataset}_{outcome}_IRM_"
                f"{'clustered' if specification == 'clustered' else 'iid'}"
            )
            selected = weights[
                (weights["model"] == model_name)
                & (
                    weights["nuisance"].eq(nuisance)
                    if nuisance != "ml_g" else
                    weights["nuisance"].astype(str).str.startswith("ml_g")
                )
            ]
            if not selected.empty:
                return selected.groupby("learner")["weight"].mean().to_dict()

        # Read compact checkpoint weights before looking for full fitted
        # nuisance learners. This keeps the publication table lightweight.
        compact = getattr(model, "convex_weights", {})
        if compact:
            if nuisance == "ml_g":
                # Multivalued outcomes may store one outcome learner per
                # treatment level (ml_g0, ml_g1, ...). Average them for the
                # single outcome-learner column in the publication table.
                rows = [
                    values for key, values in compact.items()
                    if str(key).startswith("ml_g")
                ]
                names = list(dict.fromkeys(
                    name for row in rows for name in row
                ))
                compact_weights = {
                    name: float(np.mean([
                        row[name] for row in rows if name in row
                    ]))
                    for name in names
                }
            else:
                compact_weights = compact.get(nuisance, {})
            if compact_weights:
                return {name: float(value) for name, value in compact_weights.items()}

        models = model.models or {}
        fitted = (
            {key: value for key, value in models.items() if key.startswith("ml_g")}
            if nuisance == "ml_g" else models.get(nuisance, {})
        )
        rows = []

        def visit(value):
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, np.ndarray):
                for item in value.flat:
                    visit(item)
            elif hasattr(value, "weights_"):
                names = [name for name, _ in value.estimators]
                rows.append(dict(zip(names, np.asarray(value.weights_))))

        visit(fitted)
        if not rows:
            raise ValueError(f"No stored weights found for nuisance learner {nuisance}.")
        names = list(dict.fromkeys(name for row in rows for name in row))
        return {
            name: float(np.mean([row[name] for row in rows if name in row]))
            for name in names
        }

    learner_names = {
        "ml_g": ["ols", "lasso", "elastic_net", "random_forest", "xgboost"],
        "ml_m": ["logit", "lasso", "elastic_net", "random_forest", "xgboost"],
    }
    labels = {
        "SomeRiskHome": "Any detectable E. coli at home",
        "VeryHighRiskHome": "Very high E. coli at home (>100 CFU/100 mL)",
        "diarrhea": "Diarrhea among children under five",
    }
    output_dir = Path(output_dir)
    paths = []
    for specification, suffix in [("clustered", "cluster"), ("unclustered", "no_cluster")]:
        lines = [
            r"% Requires: \usepackage{booktabs}",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Super Learner weights: {specification} folds}}",
            rf"\label{{tab:super-learner-weights-{specification}}}",
            r"\small",
            r"\begin{tabular}{lrlr}",
            r"\toprule",
            r"Learner $g(X)$ & Weight & Learner $m(X)$ & Weight \\",
            r"\midrule",
        ]
        for panel_index, outcome in enumerate(outcome_order):
            dataset = "U5" if outcome == "diarrhea" else "HH"
            model = None if weights is not None and not weights.empty else estimates[(dataset, outcome)][f"irm_{suffix}"]
            g_weights = collect_weights(
                model, "ml_g", dataset, outcome, specification
            )
            m_weights = collect_weights(
                model, "ml_m", dataset, outcome, specification
            )
            lines.append(
                rf"\multicolumn{{4}}{{l}}{{\textit{{Panel {'ABC'[panel_index]}: {labels[outcome]}}}}} \\"
            )
            for g_name, m_name in zip(learner_names["ml_g"], learner_names["ml_m"]):
                g_value = "---" if g_name not in g_weights else f"{g_weights[g_name]:.3f}"
                m_value = "---" if m_name not in m_weights else f"{m_weights[m_name]:.3f}"
                lines.append(
                    f"{_latex_text(g_name)} & {g_value} & "
                    f"{_latex_text(m_name)} & {m_value} " + r"\\"
                )
            if panel_index < len(outcome_order) - 1:
                lines.append(r"\midrule")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            r"\begin{minipage}{0.9\linewidth}\footnotesize Weights are averaged across outer folds and repetitions. The $g(X)$ learner predicts the outcome; the $m(X)$ learner predicts treatment assignment.\end{minipage}",
            r"\end{table}",
        ]
        path = output_dir / f"table_super_learner_weights_{specification}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_publication_table(
    output_dir,
    estimates,
    outcome_order,
    filename,
    caption,
    label,
    report_levels,
    folds,
    repetitions,
    specifications=("clustered",),
) -> Path:
    """Write a regression-style table for selected stacked specifications.

    Household E. coli outcomes are grouped in one table and diarrhea is passed
    separately.  The main-paper default is the clustered-fold specification.
    The appendix can request both ``clustered`` and ``unclustered``. APOS
    cluster-fold SEs use the custom one-way cluster sandwich computed from the
    contrast influence scores.
    """

    return _write_column_publication_table(
        output_dir, estimates, outcome_order, filename, caption, label,
        report_levels, folds, repetitions, specifications
    )

    valid_specifications = {"clustered", "unclustered"}
    if not specifications or not set(specifications).issubset(valid_specifications):
        raise ValueError("specifications must contain clustered and/or unclustered.")

    model_suffix = {
        "clustered": "cluster",
        "unclustered": "no_cluster",
    }
    specification_labels = {
        "clustered": "Clustered folds",
        "unclustered": "Ordinary folds",
    }

    def result_cell(summary, row_number=0, se_override=None):
        row = summary.iloc[row_number]
        se = float(row["std err"]) if se_override is None else float(se_override)
        return (
            _format_coef(float(row["coef"]), se),
            _format_se(se),
        )

    def sample_statistics(frame, outcome, treatment):
        """Return descriptive sample statistics for one fitted specification."""

        shares = frame[treatment].value_counts(normalize=True).reindex(
            [0, 1, 2, 3, 98], fill_value=0.0
        )
        return {
            "n": len(frame),
            "psu": frame["Cluster_var"].nunique()
            if "Cluster_var" in frame else None,
            "mean_y": float(frame.loc[frame[treatment].eq(0), outcome].mean()),
            "mean_d": float(frame[treatment].mean())
            if treatment == "water_treatment" else None,
            "shares": shares,
        }

    def share_text(value):
        """Format an APOS treatment share as a LaTeX percentage."""

        return f"{100 * float(value):.1f}\\%"

    lines = [
        r"% Requires: \usepackage{booktabs, graphicx, adjustbox}",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{l{'c' * len(specifications)}}}",
        r"\hline\hline",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{({index})}}"
            for index, _ in enumerate(specifications, start=1)
        ) + " " + chr(92) * 2,
        "Outcome / effect & " + " & ".join(
            specification_labels[specification] for specification in specifications
        ) + " " + chr(92) * 2,
        r"\hline",
    ]

    for outcome_index, outcome in enumerate(outcome_order):
        dataset_name = "U5" if outcome == "diarrhea" else "HH"
        models = estimates[(dataset_name, outcome)]
        irm_summaries = [
            models[f"irm_{model_suffix[specification]}"].summary
            for specification in specifications
        ]
        apos_summaries = [
            models[f"apos_{model_suffix[specification]}"].causal_contrast(
                reference_levels=[0]
            ).summary.iloc[:len(report_levels) - 1]
            for specification in specifications
        ]
        apos_cluster = models["apos_cluster"].causal_contrast(
            reference_levels=[0]
        ).summary.iloc[:len(report_levels) - 1]

        label = _get_outcome_label(outcome)
        panel_label = f"Panel {chr(65 + outcome_index)}: {label}"
        lines.append(
            rf"\multicolumn{{{1 + len(specifications)}}}{{l}}{{\textbf{{{panel_label}}}}} \\")

        irm_cells = [
            result_cell(summary)
            for summary in irm_summaries
        ]
        def psu_text(stats):
            """Format PSU counts, unavailable for unclustered fits."""

            return "---" if stats["psu"] is None else f"{stats['psu']:,}"

        lines.extend([
            r"\quad Any treatment (IRM) & " + " & ".join(
                coefficient for coefficient, _ in irm_cells
            ) + " " + chr(92) * 2,
            r" & " + " & ".join(
                standard_error for _, standard_error in irm_cells
            ) + " " + chr(92) * 2,
        ])

        for row_number, contrast in enumerate(apos_summaries[0].index):
            apos_cells = []
            for specification, summary in zip(specifications, apos_summaries):
                se_override = None
                if specification == "clustered":
                    se_override = models.get("apos_cluster_se", [None] * len(summary))[row_number]
                apos_cells.append(result_cell(summary, row_number, se_override))
            lines.extend([
                rf"\quad {contrast} (APOS) & " + " & ".join(
                    coefficient for coefficient, _ in apos_cells
                ) + " " + chr(92) * 2,
                r" & " + " & ".join(
                    standard_error for _, standard_error in apos_cells
                ) + " " + chr(92) * 2,
            ])

        frame_map = {
            "clustered": ("frame_cluster",),
            "unclustered": ("frame_no_cluster",),
        }
        irm_stats = [
            sample_statistics(models[f"irm_{frame_map[specification][0]}"], outcome, "water_treatment")
            for specification in specifications
        ]
        apos_stats = [
            sample_statistics(models[f"apos_{frame_map[specification][0]}"], outcome, "WQ15_g")
            for specification in specifications
        ]

        def stat_row(label, *values):
            """Format a statistics row for one or two specifications."""

            # Use chr(92) to guarantee exactly two LaTeX row terminators.
            return f"{label} & " + " & ".join(values) + " " + chr(92) * 2

            return f"{label} & " + " & ".join(values) + " \\\\\\" 

        lines.extend([
            stat_row(r"\quad \textit{Shares} (IRM)", *([""] * len(irm_stats))),
            stat_row(r"\quad Outcome mean (\%)", *[share_text(stats["mean_y"]) for stats in irm_stats]),
            stat_row(r"\quad Treated (\%) (IRM)", *[share_text(stats["mean_d"]) for stats in irm_stats]),
            stat_row(r"\quad Observations (IRM)", *[f"{stats['n']:,}" for stats in irm_stats]),
            stat_row(r"\quad PSUs (IRM)", *[psu_text(stats) for stats in irm_stats]),
            stat_row(r"\quad \textit{Shares} (APOS)", *([""] * len(apos_stats))),
            stat_row(r"\quad Observations (APOS)", *[f"{stats['n']:,}" for stats in apos_stats]),
            stat_row(r"\quad PSUs (APOS)", *[psu_text(stats) for stats in apos_stats]),
            *[
                stat_row(
                    rf"\quad {_SHARE_LABELS[level]} (APOS)",
                    *[share_text(stats["shares"].loc[level]) for stats in apos_stats],
                )
                for level in [0, 1, 2, 3, 98]
            ],
            stat_row(r"\quad Controls", *(["Yes"] * len(specifications))),
            r"\addlinespace",
        ])

        if outcome_index < len(outcome_order) - 1:
            lines.append(r"\hline")

    notes = (
        r"\textit{Notes:} Cells report coefficients with significance stars and "
        r"standard errors in parentheses. Clustered specifications use "
        r"cluster-level sample splitting and cluster-robust IRM inference; "
        r"APOS cluster-fold standard errors use a one-way cluster sandwich on "
        r"the contrast influence scores. The unclustered specifications use ordinary "
        r"sample splitting. All models include the common household controls; "
        r"U5 models additionally include child age and sex. APOS estimates all "
        r"treatment levels, including level 98, but the table reports levels "
        r"1--3 against level 0. "
        r"Mean $D$ is reported for the binary IRM treatment; APOS reports "
        r"treatment shares because its treatment codes are categorical. "
        rf"Cross-fitting uses {folds} folds and {repetitions} repetitions. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. "
        r"A conventional OLS $R^2$ is not reported because these are DoubleML "
        r"orthogonal-score estimates; descriptive BLP $R^2$ values are reported "
        r"separately in the GATE tables."
    )
    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        rf"\begin{{minipage}}{{\linewidth}}\scriptsize {notes}\end{{minipage}}",
        r"\end{table}",
    ])

    path = Path(output_dir) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_stacked_regression_table(
    output_dir,
    estimates,
    sensitivity_results,
    outcome_order,
    filename,
    caption,
    label,
    report_levels,
    folds,
    repetitions,
    specifications=("clustered",),
):
    """Write the main or appendix stacked regression-style table."""

    # Keep the main publication layout consistent before and after sensitivity.
    # Sensitivity outputs are written separately; this table is the compact
    # outcome-by-column presentation requested for the main results.
    return _write_column_publication_table(
        output_dir, estimates, outcome_order, filename, caption, label,
        report_levels, folds, repetitions, specifications
    )

    suffix = {"clustered": "cluster", "unclustered": "no_cluster"}
    spec_label = {"clustered": "Clustered folds", "unclustered": "Ordinary folds"}
    treatments = [
        ("IRM", "Any treatment", "Any Treatment"),
        ("APOS", "Boiling", "Boiling"),
        ("APOS", "Chlorination/tablets", "Chlorination/tablets"),
        ("APOS", "Straining/settling", "Straining/settling"),
    ]
    row_end = chr(92) * 2
    frame_keys = {"clustered": "cluster", "unclustered": "no_cluster"}
    ncols = len(specifications) * len(treatments)
    lines = [
        r"% Requires: \usepackage{booktabs, graphicx, adjustbox}",
        r"\begin{table}[!ht]\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{l{'c' * ncols}}}",
        r"\hline\hline",
    ]
    for specification in specifications:
        lines.append(
            " & " + rf"\multicolumn{{4}}{{c}}{{{spec_label[specification]}}}"
            + (" &" if specification != specifications[-1] else "")
        )
    lines.append(row_end)
    lines.append(
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{({index})}}"
            for index in range(1, ncols + 1)
        ) + " " + row_end
    )
    lines.append(
        " & " + " & ".join(treatment[1] for _ in specifications for treatment in treatments)
        + " " + row_end
    )
    lines.append(r"\hline")

    sensitivity = sensitivity_results if sensitivity_results is not None else pd.DataFrame()

    def fmt_value(value):
        return "---" if value is None or pd.isna(value) else f"{float(value) * 100:.1f}\\%"

    def stats(frame, outcome, treatment):
        shares = frame[treatment].value_counts(normalize=True).reindex(
            [0, 1, 2, 3, 98], fill_value=0.0
        )
        return {
            "n": len(frame),
            "psu": frame["Cluster_var"].nunique(),
            "mean_y": float(frame.loc[frame[treatment].eq(0), outcome].mean()),
            "mean_d": float(frame[treatment].mean()) if treatment == "water_treatment" else None,
            "shares": shares,
        }

    def rv_cell(outcome, specification, method, treatment, key):
        if sensitivity.empty:
            return "---"
        hit = sensitivity[
            sensitivity["outcome"].eq(outcome)
            & sensitivity["specification"].eq(specification)
            & sensitivity["method"].eq(method)
            & sensitivity["treatment"].eq(treatment)
        ]
        if hit.empty:
            return "---"
        return fmt_value(hit.iloc[0][key])

    def nuisance_weights(model, nuisance):
        """Return base-learner names and convex weights from a fitted learner."""

        learner = model.learner[nuisance]
        names = [name for name, _ in learner.estimators]
        weights = np.asarray(learner.weights_, dtype=float)
        return list(zip(names, weights))

    for panel_index, outcome in enumerate(outcome_order):
        dataset = "U5" if outcome == "diarrhea" else "HH"
        models = estimates[(dataset, outcome)]
        panel_label = f"Panel {chr(65 + panel_index)}: {_get_outcome_label(outcome)}"
        lines.append(rf"\multicolumn{{{ncols + 1}}}{{l}}{{\textbf{{{panel_label}}}}} {row_end}")
        coefficient_rows = []
        se_rows = []
        for method, treatment_label, sensitivity_label in treatments:
            coefficients = []
            ses = []
            for specification in specifications:
                model = models[f"{method.lower()}_{suffix[specification]}"]
                summary = model.summary if method == "IRM" else model.causal_contrast(
                    reference_levels=[0]
                ).summary.iloc[:len(report_levels) - 1]
                row_number = 0 if method == "IRM" else {"Boiling": 0, "Chlorination/tablets": 1, "Straining/settling": 2}[treatment_label]
                row = summary.iloc[row_number]
                se = float(row["std err"])
                if method == "APOS" and specification == "clustered":
                    se = float(models["apos_cluster_se"][row_number])
                coefficients.append(_format_coef(float(row["coef"]), se))
                ses.append(_format_se(se))
            coefficient_rows.append((treatment_label, coefficients))
            se_rows.append((treatment_label, ses))
        for treatment_label, values in coefficient_rows:
            lines.append(rf"{treatment_label} & " + " & ".join(values * len(specifications)) + " " + row_end)
            se_values = next(values for label, values in se_rows if label == treatment_label)
            lines.append(" & " + " & ".join(se_values * len(specifications)) + " " + row_end)

        for key, title in [("rv_q", r"RV"), ("rv_qa", r"RV$_\alpha$")]:
            cells = []
            for specification in specifications:
                cells.append(rv_cell(outcome, specification, "IRM", "Any Treatment", key))
                for treatment in ["Boiling", "Chlorination/tablets", "Straining/settling"]:
                    cells.append(rv_cell(outcome, specification, "APOS", treatment, key))
            lines.append(rf"\quad {title} & " + " & ".join(cells) + " " + row_end)

        for stat_label, method, treatment_column in [
            (r"\quad \textit{Shares} (IRM)", "IRM", "water_treatment"),
            (r"\quad Outcome mean (\%)", "IRM", "water_treatment"),
            (r"\quad Mean treatment (IRM)", "IRM", "water_treatment"),
            (r"\quad Observations (IRM)", "IRM", "water_treatment"),
            (r"\quad PSUs (IRM)", "IRM", "water_treatment"),
            (r"\quad \textit{Shares} (APOS)", "APOS", "WQ15_g"),
            (r"\quad Outcome mean (\%)", "APOS", "WQ15_g"),
            (r"\quad Observations (APOS)", "APOS", "WQ15_g"),
            (r"\quad PSUs (APOS)", "APOS", "WQ15_g"),
        ]:
            cells = []
            for specification in specifications:
                frame = models[f"{method.lower()}_frame_{frame_keys[specification]}"]
                stat = stats(frame, outcome, treatment_column)
                if stat_label == r"\quad Outcome mean (\%)":
                    value = share_text(stat["mean_y"])
                elif stat_label.startswith(r"\quad \textit{Shares"):
                    value = ""
                elif stat_label == r"\quad Mean treatment (IRM)":
                    value = fmt_value(stat["mean_d"])
                elif stat_label.startswith(r"\quad Observations"):
                    value = f"{stat['n']:,}"
                else:
                    value = f"{stat['psu']:,}"
                cells.extend([value] * len(treatments))
            lines.append(stat_label + " & " + " & ".join(cells) + " " + row_end)
        for level in [0, 1, 2, 3, 98]:
            cells = []
            for specification in specifications:
                stat = stats(models[f"apos_frame_{frame_keys[specification]}"], outcome, "WQ15_g")
                cells.extend([fmt_value(stat["shares"].loc[level])] * len(treatments))
            lines.append(rf"\quad {_SHARE_LABELS[level]} (APOS) & " + " & ".join(cells) + " " + row_end)

        if len(specifications) > 1:
            lines.append(rf"\multicolumn{{{ncols + 1}}}{{l}}{{\textit{{Nuisance learner weights}}}} {row_end}")
            for nuisance, nuisance_label in [("ml_g", r"$g(X)$"), ("ml_m", r"$m(X)$")]:
                lines.append(rf"\multicolumn{{{ncols + 1}}}{{l}}{{\quad Nuisance {nuisance_label}}} {row_end}")
                learner_names = []
                for specification in specifications:
                    for method, _, _ in treatments:
                        model = models[f"{method.lower()}_{suffix[specification]}"]
                        learner_names.extend(name for name, _ in nuisance_weights(model, nuisance))
                for learner_name in dict.fromkeys(learner_names):
                    cells = []
                    for specification in specifications:
                        for method, _, _ in treatments:
                            model = models[f"{method.lower()}_{suffix[specification]}"]
                            weights = dict(nuisance_weights(model, nuisance))
                            cells.append(f"{weights.get(learner_name, np.nan):.3f}" if learner_name in weights else "---")
                    lines.append(rf"\quad {learner_name} & " + " & ".join(cells) + " " + row_end)
        if panel_index < len(outcome_order) - 1:
            lines.append(r"\hline")

    notes = (
        r"\textit{Notes:} Coefficients are from the stacked specification. "
        r"Clustered-fold APOS standard errors are the custom one-way cluster "
        r"sandwich computed from the APOS contrast influence scores; clustered "
        r"IRM standard errors use the cluster-robust influence-score sandwich. "
        rf"Cross-fitting uses {folds} folds and {repetitions} repetitions. "
        r"Other treatment is omitted from the displayed APOS contrasts and shares. "
        r"RV and RV$_\alpha$ are the original, not relative, robustness values."
    )
    lines += [
        r"\hline\hline", r"\end{tabular}", r"\end{adjustbox}",
        rf"\par\vspace{{3pt}}\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
        r"\end{table}",
    ]
    path = Path(output_dir) / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _create_heterogeneity_comparison_tables_legacy(
    results,
    output_dir=None,
    filename_prefix="table_heterogeneity",
    outcome_sets=(
        ("ecoli", ["SomeRiskHome", "VeryHighRiskHome"]),
        ("diarrhea", ["diarrhea"]),
    ),
):
    """Write outcome-specific tables for clustered and ordinary GATE results.

    Rows are heterogeneity-group categories. Each treatment cell reports the
    cluster-fold estimate and SE, followed in brackets by the ordinary-fold
    estimate and SE. Panel footers report sample sizes, PSU counts, and
    descriptive BLP R-squared values.
    """

    import pandas as pd

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    frame = frame[frame["specification"].isin(["clustered_folds", "unclustered"])]
    if frame.empty:
        raise ValueError("No heterogeneity results remain for comparison tables.")

    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    treatment_order = [
        ("IRM", "Any Treatment"),
        ("APOS", "Boiling"),
        ("APOS", "Chlorination/tablets"),
        ("APOS", "Straining/settling"),
    ]
    paths = []

    for table_name, outcomes in outcome_sets:
        table_frame = frame[frame["outcome"].isin(outcomes)].copy()
        if table_frame.empty:
            continue
        n_effect_columns = len(outcomes) * len(treatment_order)
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}",
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Heterogeneity of stacked GATE effects: {table_name}}}",
            rf"\label{{tab:{filename_prefix}-{table_name}}}",
            r"\scriptsize\setlength{\tabcolsep}{3pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{l{'c' * n_effect_columns}}}",
            r"\toprule",
            " & ".join(
                ["Heterogeneity group"]
                + [
                    rf"\multicolumn{{{len(treatment_order)}}}{{c}}{{{_get_outcome_label(outcome)}}}"
                    for outcome in outcomes
                ]
            ) + r" \\",
            " & ".join(
                [""]
                + [treatment for _ in outcomes for _, treatment in treatment_order]
            ) + r" \\",
            r"\midrule",
        ]

        panels = list(dict.fromkeys(table_frame["group"].tolist()))
        for panel_index, group in enumerate(panels):
            panel = table_frame[table_frame["group"].eq(group)]
            panel_label = panel["heterogeneity_label"].iloc[0]
            n_columns = 1 + n_effect_columns
            lines.append(
                rf"\multicolumn{{{n_columns}}}{{l}}"
                rf"{{\textit{{Panel {chr(65 + panel_index)}: {panel_label}}}}} \\"
            )
            values = list(dict.fromkeys(panel["group_value"].tolist()))
            values.sort(key=str)

            for raw_value in values:
                row = panel[panel["group_value"].eq(raw_value)]
                cells = [str(row["group_label"].iloc[0])]
                for outcome in outcomes:
                    for method, treatment in treatment_order:
                        hit = row[
                            row["outcome"].eq(outcome)
                            & row["method"].eq(f"{method} stacked")
                            & row["treatment_label"].eq(treatment)
                        ]
                        clustered = hit[hit["specification"].eq("clustered_folds")]
                        ordinary = hit[hit["specification"].eq("unclustered")]
                        if clustered.empty or ordinary.empty:
                            cells.append("---")
                            continue
                        c = clustered.iloc[0]
                        o = ordinary.iloc[0]
                        cells.append(
                            f"{_format_coef(float(c['coef']), float(c['se']))}"
                            f" ({float(c['se']):.3f}) "
                            f"[{_format_coef(float(o['coef']), float(o['se']))}"
                            f" ({float(o['se']):.3f})]"
                        )
                lines.append(" & ".join(cells) + r" \\")

            # Regression-style diagnostics are reported in every heterogeneity panel.
            for outcome in outcomes:
                for method in ["IRM", "APOS"]:
                    hit = panel[
                        panel["outcome"].eq(outcome)
                        & panel["method"].eq(f"{method} stacked")
                    ]
                    if hit.empty:
                        continue
                    clustered_hit = hit[hit["specification"].eq("clustered_folds")]
                    clustered_hit = clustered_hit.drop_duplicates(["group_value"])
                    n_obs = int(clustered_hit["n"].sum())
                    n_psu = (
                        int(clustered_hit["n_psu"].sum())
                        if clustered_hit["n_psu"].notna().all() else "---"
                    )
                    lines.append(
                        rf"\multicolumn{{{n_columns}}}{{l}}"
                        rf"{{\quad $N$ ({outcome}, {method}) = {n_obs:,}; "
                        rf"PSUs = {n_psu}}} \\"
                    )
                    for treatment in [t for m, t in treatment_order if m == method]:
                        treatment_hit = hit[hit["treatment_label"].eq(treatment)]
                        if treatment_hit.empty:
                            continue
                        c = treatment_hit[treatment_hit["specification"].eq("clustered_folds")]
                        o = treatment_hit[treatment_hit["specification"].eq("unclustered")]
                        if c.empty or o.empty:
                            continue
                        r2_cell = f"{float(c['r2'].iloc[0]):.3f} [{float(o['r2'].iloc[0]):.3f}]"
                        lines.append(
                            rf"\multicolumn{{{n_columns}}}{{l}}"
                            rf"{{\quad BLP $R^2$ ({outcome}, {method}, {treatment}) = "
                            rf"{r2_cell}}} \\"
                        )
            if panel_index < len(panels) - 1:
                lines.append(r"\midrule")

        notes = (
            r"\textit{Notes:} Each cell reports the clustered-fold GATE estimate "
            r"with its SE, followed in brackets by the ordinary-fold estimate "
            r"and SE. $N$ and PSU counts are reported for each outcome and "
            r"method in every heterogeneity panel. BLP $R^2$ is descriptive: it is "
            r"the R-squared from projecting the APOS/IRM orthogonal contrast "
            r"signal onto heterogeneity-group indicators, not a causal-model $R^2$. "
            r"Country is exploratory."
        )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}",
            r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{table_name}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")

    return paths


def create_heterogeneity_comparison_tables(
    results,
    output_dir=None,
    filename_prefix="table_heterogeneity",
    specifications=("clustered_folds", "unclustered"),
    outcome_sets=(
        ("ecoli", ["SomeRiskHome", "VeryHighRiskHome"]),
        ("diarrhea", ["diarrhea"]),
    ),
):
    """Write heterogeneity tables in the water-treatment appendix layout."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    frame = frame[frame["specification"].isin(specifications)]
    if frame.empty:
        raise ValueError("No heterogeneity results remain for comparison tables.")

    output_dir = Path(output_dir) if output_dir is not None else Path("Output")
    output_dir.mkdir(parents=True, exist_ok=True)
    treatment_order = [
        ("IRM", "Any Treatment"),
        ("APOS", "Boiling"),
        ("APOS", "Chlorination/tablets"),
        ("APOS", "Straining/settling"),
    ]
    paths = []

    def sort_value(value):
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    for table_name, outcomes in outcome_sets:
        table_frame = frame[frame["outcome"].isin(outcomes)].copy()
        if table_frame.empty:
            continue
        n_effect_columns = len(specifications) * len(treatment_order)
        n_columns = 1 + n_effect_columns
        if len(specifications) == 1:
            treatment_header = "Heterogeneity group & " + " & ".join(
                treatment for _, treatment in treatment_order
            ) + r" \\"
            fold_header = None
        else:
            treatment_header = "Heterogeneity group & " + " & ".join(
                rf"\multicolumn{{{len(specifications)}}}{{c}}{{{treatment}}}"
                for _, treatment in treatment_order
            ) + r" \\"
            fold_header = " & " + " & ".join(
                [
                    {"clustered_folds": "Clustered folds", "unclustered": "Ordinary folds"}.get(
                        specification, specification
                    )
                    for _ in treatment_order for specification in specifications
                ]
            ) + r" \\"
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
            r"\begin{landscape}", r"\begin{table}[p]", r"\centering",
            rf"\caption{{Heterogeneity of stacked GATE effects: {table_name}}}",
            rf"\label{{tab:{filename_prefix}-{table_name}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{l{'c' * n_effect_columns}}}",
            r"\toprule",
            treatment_header,
            r"\midrule",
        ]
        if fold_header is not None:
            lines.insert(-1, fold_header)

        for panel_index, outcome in enumerate(outcomes):
            outcome_frame = table_frame[table_frame["outcome"].eq(outcome)]
            if outcome_frame.empty:
                continue
            groups = list(dict.fromkeys(outcome_frame["group"].tolist()))
            for group_index, group in enumerate(groups):
                panel = outcome_frame[outcome_frame["group"].eq(group)]
                group_label = panel["heterogeneity_label"].iloc[0]
                panel_label = f"{_get_outcome_label(outcome)}: {group_label}"
                lines.append(
                    rf"\multicolumn{{{n_columns}}}{{l}}{{\textbf{{Panel {chr(65 + panel_index)}: {panel_label}}}}} \\")
                values = sorted(
                    dict.fromkeys(panel["group_value"].tolist()),
                    key=sort_value,
                )

                for raw_value in values:
                    row = panel[panel["group_value"].eq(raw_value)]
                    estimate_cells, se_cells = [], []
                    for method, treatment in treatment_order:
                        hit = row[
                            row["method"].eq(f"{method} stacked")
                            & row["treatment_label"].eq(treatment)
                        ]
                        for specification in specifications:
                            result = hit[hit["specification"].eq(specification)]
                            if result.empty:
                                estimate_cells.append("---")
                                se_cells.append("---")
                            else:
                                result = result.iloc[0]
                                estimate_cells.append(
                                    _format_coef(float(result["coef"]), float(result["se"]))
                                )
                                se_cells.append(f"({float(result['se']):.3f})")
                    label = str(row["group_label"].iloc[0])
                    lines.append(" & ".join([label] + estimate_cells) + r" \\")
                    lines.append(" & ".join([""] + se_cells) + r" \\")

                lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{Sample}}}} \\")
                sample_stats = {}
                for method in ("IRM", "APOS"):
                    hit = panel[panel["method"].eq(f"{method} stacked")]
                    if hit.empty:
                        continue
                    sample_specification = (
                        "clustered_folds" if "clustered_folds" in specifications
                        else specifications[0]
                    )
                    sample = hit[hit["specification"].eq(sample_specification)]
                    sample = sample.drop_duplicates(["group_value"])
                    n_obs = int(sample["n"].sum())
                    n_psu = (
                        int(sample["n_psu"].sum())
                        if sample["n_psu"].notna().all() else "---"
                    )
                    sample_stats[method] = {
                        "N": f"{n_obs:,}",
                        "PSUs": f"{n_psu:,}" if isinstance(n_psu, int) else str(n_psu),
                    }

                # N and PSU counts are sample-level diagnostics.  Put each
                # method's count only in the treatment columns that use that
                # method, without adding IRM/APOS labels as pseudo-columns.
                for statistic in ("N", "PSUs"):
                    cells = [rf"\quad {statistic}"]
                    for method, _ in treatment_order:
                        value = sample_stats.get(method, {}).get(statistic, "")
                        cells.extend([value] * len(specifications))
                    lines.append(" & ".join(cells) + r" \\")

                lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{BLP $R^{{2}}$}}}} \\")
                for method in ("IRM", "APOS"):
                    hit = panel[panel["method"].eq(f"{method} stacked")]
                    if hit.empty:
                        continue
                    for treatment in [t for m, t in treatment_order if m == method]:
                        treatment_hit = hit[hit["treatment_label"].eq(treatment)]
                        available = [
                            treatment_hit[treatment_hit["specification"].eq(specification)]
                            for specification in specifications
                        ]
                        if any(value.empty for value in available):
                            continue
                        # Keep each BLP R-squared under its corresponding
                        # treatment column.  Spanning the whole row makes the
                        # value appear detached from the treatment it describes.
                        r2_cells = []
                        for value in available:
                            r2_cells.append(f"{float(value['r2'].iloc[0]):.3f}")
                        cells = [rf"\quad {treatment}"]
                        for candidate_method, candidate_treatment in treatment_order:
                            if (candidate_method, candidate_treatment) == (method, treatment):
                                cells.extend(r2_cells)
                            else:
                                cells.extend([""] * len(specifications))
                        lines.append(" & ".join(cells) + r" \\")
                if group_index < len(groups) - 1:
                    lines.append(r"\addlinespace[4pt]")
            if panel_index < len(outcomes) - 1:
                lines.append(r"\midrule")

        fold_note = (
            r"Clustered folds use cluster-level sample splitting."
            if specifications == ("clustered_folds",)
            else r"Clustered folds use cluster-level sample splitting; ordinary folds use unclustered folds."
        )
        notes = (
            r"\textit{Notes:} Coefficient rows report GATE estimates with "
            r"significance stars; the following row reports standard errors in "
            r"parentheses. "
            + fold_note +
            r" $N$ and PSU counts "
            r"are reported for each outcome and method. BLP $R^2$ is descriptive: "
            r"it is the R-squared from projecting the APOS/IRM orthogonal contrast "
            r"signal onto heterogeneity-group indicators, not a causal-model $R^2$. "
            r"The heterogeneity analysis is exploratory."
        )
        lines += [
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}", r"\end{landscape}",
        ]
        path = output_dir / f"{filename_prefix}_{table_name}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"Table saved to: {path}")
    return paths


# Backward-compatible import name for cached estimation scripts.  New calls
# should use ``write_publication_table``.
write_water_treatment_table = write_publication_table
