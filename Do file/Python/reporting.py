"""Publication, sensitivity, and heterogeneity reporting boundaries."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Callable, Protocol
import warnings

import numpy as np
import pandas as pd

from ddml import (
    cluster_robust_framework_inference,
    format_coefficient,
    summary_with_clustered_inference,
)


class ReportingSpec(Protocol):
    """Minimal callbacks needed by the shared reporting entry points."""

    effect_writer: Callable[..., Any]
    sensitivity_runner: Callable[..., Any]
    gate_runner: Callable[..., Any]
    manifest_writer: Callable[..., Path]


def _fold_modes(fold_mode: str) -> tuple[str, ...]:
    choices = {
        "clustered": ("clustered",),
        "unclustered": ("unclustered",),
        "both": ("clustered", "unclustered"),
        "all": ("clustered", "unclustered"),
    }
    try:
        return choices[fold_mode]
    except KeyError as error:
        raise ValueError(f"Unknown fold mode: {fold_mode}") from error


def save_effect_outputs(
    spec: ReportingSpec,
    estimates,
    *,
    quick_sample,
    file_suffix="",
    fold_mode="both",
):
    """Write effect outputs using only the requested fold bundles."""
    return spec.effect_writer(
        estimates,
        quick_sample=quick_sample,
        file_suffix=file_suffix,
        fold_modes=_fold_modes(fold_mode),
    )


def run_sensitivity(spec: ReportingSpec, estimates, *, quick_sample, fold_mode="both"):
    """Run sensitivity reporting for the requested fold bundles."""
    return spec.sensitivity_runner(
        estimates,
        quick_sample=quick_sample,
        fold_modes=_fold_modes(fold_mode),
    )


def run_gate(spec: ReportingSpec, estimates, *, quick_sample, fold_mode="both"):
    """Run GATE reporting for the requested fold bundles."""
    return spec.gate_runner(
        estimates,
        quick_sample=quick_sample,
        fold_modes=_fold_modes(fold_mode),
    )


def write_manifest(
    spec: ReportingSpec,
    *,
    model_provenance,
    sensitivity_provenance,
    fold_mode,
):
    """Write the current analysis manifest through the configured writer."""
    return spec.manifest_writer(
        model_provenance=model_provenance,
        sensitivity_provenance=sensitivity_provenance,
        fold_mode=fold_mode,
    )


BENCHMARK_GROUPS = (
    ("wealth_index", ("windex5",)),
    ("urban", ("urban",)),
    ("water_source", ("WS1_g",)),
    ("source_ecoli", ("wq27_decile",)),
    ("under_five", ("Any_U5",)),
    ("household_children", ("Girls_less_than15", "Boys_15or_less")),
    ("toilet", ("Toilet",)),
    ("country_fixed_effects", ("country_cat",)),
    ("child_demographics", ("age", "male")),
)


def benchmark_groups(x_columns):
    """Return complete prespecified encoded covariate blocks."""
    x_columns = tuple(x_columns)
    child = "age" in x_columns or "male" in x_columns
    result = {}
    for name, variables in BENCHMARK_GROUPS:
        if name == "child_demographics" and not child:
            continue
        columns = [
            column
            for column in x_columns
            if any(
                column == variable or column.startswith(variable + "_")
                for variable in variables
            )
        ]
        for variable in variables:
            if not any(
                column == variable or column.startswith(variable + "_")
                for column in columns
            ):
                raise ValueError(
                    f"Missing benchmark control {variable!r} in {name!r}"
                )
        result[name] = columns
    unique = {column for columns in result.values() for column in columns}
    if len(unique) != sum(map(len, result.values())):
        raise ValueError("Overlapping covariate benchmark groups")
    return result


def diagonal_equivalent(cf_y, cf_d, rho=1.0):
    """Return equal confounding strength with the same point-bound bias."""
    values = (cf_y, cf_d, rho)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Benchmark sensitivity parameters must be finite")
    if cf_y < 0 or not 0 <= cf_d <= 1 or abs(rho) > 1:
        raise ValueError("Expected cf_y >= 0, 0 <= cf_d <= 1, |rho| <= 1")
    if cf_d == 1:
        return 1.0 if cf_y > 0 and rho != 0 else 0.0
    strength = abs(rho) * math.sqrt(cf_y * cf_d / (1 - cf_d))
    if strength == 0:
        return 0.0
    return 2 * strength / (strength + math.sqrt(strength * strength + 4))


def benchmark_diagonal_equivalent(cf_y, cf_d, rho=1.0, *, label="benchmark"):
    """Convert benchmark output while retaining inadmissible raw values."""
    try:
        return diagonal_equivalent(float(cf_y), float(cf_d), float(rho))
    except (TypeError, ValueError, OverflowError) as exc:
        warnings.warn(
            f"{label}: cannot compute diagonal-equivalent sensitivity "
            f"(cf_y={cf_y!r}, cf_d={cf_d!r}, rho={rho!r}): {exc}. "
            "The raw benchmark is retained and the derived value is NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")


OUTCOME_LABELS = {
    "SomeRiskHome": r"Some Risk Home (E.coli $>0$ CFU)",
    "VeryHighRiskHome": r"Very High Risk Home (E.coli $>100$ CFU)",
    "diarrhea": "Diarrhea (under-5)",
}
TREATMENT_LABELS = {
    0: "No treatment",
    1: "Boiling",
    2: "Chlorination/tablets",
    3: "Straining/settling",
    98: "Other treatment",
}


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
    *,
    estimand,
    treatment_levels,
    treatment_labels,
):
    """Write the main IRM/APOS table with outcomes arranged in columns.

    Parameters
    ----------
    output_dir : path-like
        Destination folder.
    estimates : dict
        On-demand outcome bundles from ``estimate_all_models``.
    outcome_order : list[str]
        Left-to-right outcome order.
    filename, caption, label : str
        LaTeX filename and table metadata.
    report_levels : iterable
        Treatment levels represented in the table.
    folds, repetitions : int
        Cross-fitting choices printed in the notes.
    specifications : tuple[str, ...]
        ``clustered`` and/or ``unclustered`` columns.

    Returns
    -------
    pathlib.Path
        Path of the written ``.tex`` file.
    """

    short_outcome_labels = {
        "SomeRiskHome": "Some risk",
        "VeryHighRiskHome": "Very high risk",
        "diarrhea": "Diarrhea (U5)",
    }
    short_specification_labels = {
        "clustered": "Clustered",
        "unclustered": "Unclustered",
    }

    def sample_statistics(frame, outcome, treatment):
        """Summarize sample size, PSUs, outcome mean, and treatment shares."""

        treatment_shares = frame[treatment].value_counts(
            normalize=True
        ).reindex(treatment_levels, fill_value=0.0)
        no_treatment = frame[treatment].eq(0)
        cluster_count = None
        if "Cluster_var" in frame.columns:
            cluster_count = frame["Cluster_var"].nunique()
        treatment_mean = None
        if treatment == "water_treatment":
            treatment_mean = float(frame[treatment].mean())
        return {
            "n": len(frame),
            "clusters": cluster_count,
            "outcome_mean": float(frame.loc[no_treatment, outcome].mean()),
            "treatment_mean": treatment_mean,
            "treatment_shares": treatment_shares,
        }

    def result_cell(summary, row_number, standard_error_override=None):
        """Format one coefficient/SE pair from a selected summary row."""

        result = summary.iloc[row_number]
        standard_error = float(result["std err"])
        if standard_error_override is not None:
            standard_error = float(standard_error_override)
        coefficient = format_coefficient(
            float(result["coef"]),
            standard_error,
            p_value=float(result["P>|t|"]),
        )
        return coefficient, f"({standard_error:.3f})"

    # Build one lightweight column at a time. Models are released immediately
    # after their summaries have been copied.
    columns = []
    for outcome in outcome_order:
        dataset = "U5" if outcome == "diarrhea" else "HH"
        bundle = estimates[(dataset, outcome)]

        for specification in specifications:
            if specification == "clustered":
                irm_key = "irm_cluster"
                apos_key = "apos_cluster"
                irm_frame_key = "irm_frame_cluster"
                apos_frame_key = "apos_frame_cluster"
            else:
                irm_key = "irm_no_cluster"
                apos_key = "apos_no_cluster"
                irm_frame_key = "irm_frame_no_cluster"
                apos_frame_key = "apos_frame_no_cluster"

            irm_model = bundle[irm_key]
            irm_summary = irm_model.summary.copy()
            irm_model = None
            gc.collect()

            apos_model = bundle[apos_key]
            apos_contrast = apos_model.causal_contrast(
                reference_levels=[0]
            )
            apos_summary = apos_contrast.summary.copy()
            if specification == "clustered":
                apos_inference = cluster_robust_framework_inference(
                    apos_contrast,
                    bundle[apos_frame_key]["Cluster_var"].to_numpy(),
                )
                apos_summary = summary_with_clustered_inference(
                    apos_summary,
                    apos_inference,
                )
            apos_summary = apos_summary.iloc[
                :len(report_levels) - 1
            ].copy()
            apos_model = None
            apos_contrast = None
            gc.collect()

            columns.append({
                "outcome": outcome,
                "specification": specification,
                "irm_summary": irm_summary,
                "apos_summary": apos_summary,
                "irm_sample": sample_statistics(
                    bundle[irm_frame_key],
                    outcome,
                    "water_treatment",
                ),
                "apos_sample": sample_statistics(
                    bundle[apos_frame_key],
                    outcome,
                    "WQ15_g",
                ),
            })

    number_of_columns = len(columns)
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
        rf"\begin{{tabular}}{{l{'c' * number_of_columns}}}",
        r"\hline\hline",
    ]

    if len(specifications) == 1:
        outcome_header = [
            short_outcome_labels[column["outcome"]]
            for column in columns
        ]
        lines.append(" & " + " & ".join(outcome_header) + r" \\")
    else:
        outcome_header = [""]
        for outcome in outcome_order:
            outcome_header.append(
                rf"\multicolumn{{{len(specifications)}}}{{c}}"
                rf"{{{short_outcome_labels[outcome]}}}"
            )
        lines.append(" & ".join(outcome_header) + r" \\")
        fold_header = [
            short_specification_labels[column["specification"]]
            for column in columns
        ]
        lines.append(" & " + " & ".join(fold_header) + r" \\")

    lines.append(r"\hline")

    def table_row(row_label, values):
        """Join one readable label and its cells into a LaTeX table row."""

        return row_label + " & " + " & ".join(values) + r" \\"

    def show_once_per_outcome(values):
        """Blank duplicate descriptive values in two-specification tables."""

        if len(specifications) == 1:
            return values
        displayed_outcomes = set()
        displayed_values = []
        for column, value in zip(columns, values):
            outcome = column["outcome"]
            if outcome in displayed_outcomes:
                displayed_values.append("")
            else:
                displayed_values.append(value)
                displayed_outcomes.add(outcome)
        return displayed_values

    # IRM coefficient and descriptive statistics.
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}{{\textit{{{'IRM--ATT' if estimand.upper() == "ATT" else 'IRM'}}}}} \\"
    )
    irm_cells = [
        result_cell(column["irm_summary"], 0)
        for column in columns
    ]
    lines.append(table_row(
        "Any treatment",
        [coefficient for coefficient, _ in irm_cells],
    ))
    lines.append(table_row(
        "",
        [standard_error for _, standard_error in irm_cells],
    ))
    lines.append(r"\addlinespace[6pt]")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}"
        r"{\textit{Descriptive statistics}} \\"
    )
    outcome_means = [
        f"{100 * column['irm_sample']['outcome_mean']:.1f}\\%"
        for column in columns
    ]
    treatment_means = [
        f"{100 * column['irm_sample']['treatment_mean']:.1f}\\%"
        for column in columns
    ]
    lines.append(table_row(
        "Y mean, no treatment (\\%)",
        show_once_per_outcome(outcome_means),
    ))
    lines.append(table_row(
        "Treated (\\%)",
        show_once_per_outcome(treatment_means),
    ))

    # APOS coefficients and treatment shares.
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}"
        rf"{{\textit{{{'Weighted APOS--ATT' if estimand.upper() == "ATT" else 'APOS'}}}}} \\"
    )
    reported_treatments = [
        "Boiling",
        "Chlorination/tablets",
        "Straining/settling",
    ]
    for row_number, treatment_label in enumerate(reported_treatments):
        apos_cells = []
        for column in columns:
            apos_cells.append(result_cell(
                column["apos_summary"],
                row_number,
            ))
        lines.append(table_row(
            treatment_label,
            [coefficient for coefficient, _ in apos_cells],
        ))
        lines.append(table_row(
            "",
            [standard_error for _, standard_error in apos_cells],
        ))

    lines.append(r"\addlinespace[6pt]")
    for treatment_level in treatment_levels:
        shares = [
            f"{100 * column['apos_sample']['treatment_shares'].loc[treatment_level]:.1f}\\%"
            for column in columns
        ]
        lines.append(table_row(
            treatment_labels[treatment_level],
            show_once_per_outcome(shares),
        ))

    # Sample size.
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}{{\textit{{Sample}}}} \\"
    )
    observations = [
        f"{column['irm_sample']['n']:,}"
        for column in columns
    ]
    clusters = []
    for column in columns:
        cluster_count = column["irm_sample"]["clusters"]
        clusters.append(
            "---" if cluster_count is None else f"{cluster_count:,}"
        )
    lines.append(table_row(
        "Observations",
        show_once_per_outcome(observations),
    ))
    lines.append(table_row(
        "PSUs",
        show_once_per_outcome(clusters),
    ))

    estimand_note = (
        r"folds. IRM uses the ATTE score. Weighted APOS effects compare "
        r"treatment levels 1--3 with level 0 in the population of households "
        r"that use any water treatment. "
        if estimand.upper() == "ATT"
        else r"folds. APOS effects compare treatment levels 1--3 with level 0. "
    )
    notes = (
        r"\begin{minipage}{\linewidth}\scriptsize \textit{Notes:} "
        r"Cells report coefficients with significance stars and standard "
        r"errors in parentheses. Clustered specifications use cluster-level "
        r"sample splitting; ordinary specifications use observation-level "
        + estimand_note
        + f"Cross-fitting uses {folds} folds and {repetitions} repetitions. "
        + r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$."
        + r"\end{minipage}"
    )
    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        notes,
        r"\end{table}",
    ])

    output_path = Path(output_dir) / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

def write_super_learner_weights_tables(
    output_dir,
    outcome_order,
    weights,
    checkpoint_prefix,
    file_suffix,
    caption_suffix,
    fold_mode="both",
    *,
    regressor_names,
    classifier_names,
):
    """Write one compact Super Learner weight table for each fold type.

    Parameters
    ----------
    output_dir : path-like
        Destination folder.
    outcome_order : list[str]
        Outcome display order.
    weights : pandas.DataFrame
        Averaged learner weights.
    checkpoint_prefix : str
        Main or selected-country model prefix.
    file_suffix, caption_suffix : str
        Optional selected-country output labels.

    Returns
    -------
    None
        Writes one table per fold specification.
    """

    outcome_learner_names = list(regressor_names)
    treatment_learner_names = list(classifier_names)
    short_outcome_labels = {
        "SomeRiskHome": "Any detectable E. coli at home",
        "VeryHighRiskHome": "Very high E. coli at home (>100 CFU/100 mL)",
        "diarrhea": "Diarrhea among children under five",
    }

    for specification in (("clustered", "unclustered") if fold_mode == "both" else (fold_mode,)):
        lines = [
            r"% Requires: \usepackage{booktabs}",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Super Learner weights: {specification} folds"
            rf"{caption_suffix}}}",
            rf"\label{{tab:super-learner-weights-{specification}"
            rf"{file_suffix.replace('_', '-')}}}",
            r"\small",
            r"\begin{tabular}{lrlr}",
            r"\toprule",
            r"Learner $g(X)$ & Weight & Learner $m(X)$ & Weight \\",
            r"\midrule",
        ]

        for panel_number, outcome in enumerate(outcome_order):
            dataset = "U5" if outcome == "diarrhea" else "HH"
            model_suffix = "clustered" if specification == "clustered" else "iid"
            model_name = (
                f"{checkpoint_prefix}{dataset}_{outcome}_IRM_{model_suffix}"
            )
            selected = weights[weights["model"].eq(model_name)]

            outcome_weights = selected[
                selected["nuisance"].astype(str).str.startswith("ml_g")
            ].groupby("learner")["weight"].mean()
            treatment_weights = selected[
                selected["nuisance"].eq("ml_m")
            ].groupby("learner")["weight"].mean()

            panel_letter = "ABC"[panel_number]
            panel_label = short_outcome_labels[outcome]
            lines.append(
                rf"\multicolumn{{4}}{{l}}{{\textit{{Panel "
                rf"{panel_letter}: {panel_label}}}}} \\"
            )

            paired_learners = zip(
                outcome_learner_names,
                treatment_learner_names,
            )
            for outcome_learner, treatment_learner in paired_learners:
                outcome_value = f"{outcome_weights[outcome_learner]:.3f}"
                treatment_value = f"{treatment_weights[treatment_learner]:.3f}"

                lines.append(
                    f"{outcome_learner.replace('_', r'\_')} & {outcome_value} & "
                    f"{treatment_learner.replace('_', r'\_')} & "
                    f"{treatment_value} "
                    + r"\\"
                )

            if panel_number < len(outcome_order) - 1:
                lines.append(r"\midrule")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            r"\begin{minipage}{0.9\linewidth}\footnotesize Weights are "
            r"averaged across outer folds and repetitions. The $g(X)$ learner "
            r"predicts the outcome; the $m(X)$ learner predicts treatment."
            r"\end{minipage}",
            r"\end{table}",
        ])
        output_path = (
            Path(output_dir)
            / f"table_super_learner_weights_{specification}{file_suffix}.tex"
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")

def write_sensitivity_summary_table(
    results,
    output_dir,
    filename,
    specifications,
    label,
):
    """Write a table for benchmark strength, RV, and RV-alpha.

    Parameters
    ----------
    results : pandas.DataFrame
        Completed sensitivity rows.
    output_dir : path-like
        Destination folder.
    filename : str
        LaTeX filename.
    specifications : tuple[str, ...]
        Fold specifications included as columns.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    pathlib.Path
        Path of the written table.
    """

    outcomes = ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]
    outcome_labels = {
        "SomeRiskHome": r"\shortstack{Some risk\\ at home}",
        "VeryHighRiskHome": r"\shortstack{Very high risk\\ at home}",
        "diarrhea": "Diarrhea",
    }
    specification_labels = {
        "clustered_folds": "Panel A: Clustered folds",
        "unclustered": "Panel B: Unclustered folds",
    }
    reported_effects = [
        ("IRM", "Any Treatment", "Any treatment"),
        ("APOS", "1", "Boiling"),
        ("APOS", "2", "Chlorination/tablets"),
        ("APOS", "3", "Straining/settling"),
    ]
    metrics = [
        ("r_equiv_empirical", r"$r_{\mathrm{eq}}(\hat\rho)$"),
        ("r_equiv_adversarial", r"$r_{\mathrm{eq}}(1)$"),
        ("rv", "RV"),
        ("rva", r"RV$_\alpha$"),
    ]

    number_of_result_columns = len(metrics) * len(outcomes)
    total_columns = 1 + number_of_result_columns
    if tuple(specifications) == ("clustered_folds",):
        caption = (
            "Sensitivity of DoubleML estimates to unobserved confounding"
        )
    else:
        caption = (
            "Sensitivity of DoubleML estimates: clustered and ordinary folds"
        )

    lines = [
        r"% Requires: \usepackage{booktabs, pdflscape, adjustbox}",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize\setlength{\tabcolsep}{5pt}",
        r"\begin{adjustbox}{max width=\linewidth, center}",
        rf"\begin{{tabular}}{{l{'c' * number_of_result_columns}}}",
        r"\toprule",
    ]

    outcome_header = [""]
    for outcome in outcomes:
        outcome_header.append(
            rf"\multicolumn{{{len(metrics)}}}{{c}}"
            rf"{{{outcome_labels[outcome]}}}"
        )
    lines.append(" & ".join(outcome_header) + r" \\")

    midrules = []
    for outcome_number in range(len(outcomes)):
        first_column = 2 + len(metrics) * outcome_number
        last_column = 1 + len(metrics) * (outcome_number + 1)
        midrules.append(
            rf"\cmidrule(lr){{{first_column}-{last_column}}}"
        )
    lines.append(" ".join(midrules))
    metric_labels = [label_text for _, label_text in metrics]
    lines.append(
        " & ".join(
            ["Estimand / treatment"]
            + metric_labels * len(outcomes)
        )
        + r" \\"
    )
    lines.append(r"\midrule")

    for specification_number, specification in enumerate(specifications):
        lines.append(
            rf"\multicolumn{{{total_columns}}}{{l}}"
            rf"{{\textit{{{specification_labels[specification]}}}}} \\"
        )

        previous_method = None
        for method, treatment, treatment_label in reported_effects:
            if method != previous_method:
                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    rf"{{\quad\textit{{{method}}}}} \\"
                )
                previous_method = method

            row = [rf"\qquad {treatment_label}"]
            for outcome in outcomes:
                selected = results[
                    results["specification"].eq(specification)
                    & results["method"].eq(method)
                    & results["treatment"].astype(str).eq(treatment)
                    & results["outcome"].eq(outcome)
                ]
                result = selected.iloc[0]
                for metric, _ in metrics:
                    row.append(f"{100 * float(result[metric]):.1f}\\%")
            lines.append(" & ".join(row) + r" \\")

        if specification_number < len(specifications) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{\linewidth}\footnotesize \textit{Notes:} The "
        r"Each benchmark omits an entire prespecified covariate block, including "
        r"every indicator of a categorical control or country fixed effects. "
        r"The two reported strengths are medians across blocks. "
        r"$r_{\mathrm{eq}}(\hat\rho)$ and $r_{\mathrm{eq}}(1)$ are "
        r"equal-strength, point-bias-equivalent scenarios using empirical "
        r"and maximal absolute correlation, respectively. RV is the equal strength "
        r"needed for the point bound to reach zero; RV$_\alpha$ also "
        r"includes uncertainty and is not directly decided by the two "
        r"equivalent strengths. All values are percentages. "
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ])

    output_path = Path(output_dir) / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

def create_heterogeneity_comparison_tables(
    results,
    output_dir,
    filename_prefix,
    specifications,
    include_blp_r2,
    *,
    estimand,
    outcome_labels,
):
    """Write GATE tables for household outcomes and under-five diarrhea.

    Parameters
    ----------
    results : pandas.DataFrame
        Long-form GATE estimates.
    output_dir : path-like
        Destination folder.
    filename_prefix : str
        Prefix used for household and diarrhea filenames.
    specifications : tuple[str, ...]
        Fold specifications included as columns.
    include_blp_r2 : bool
        Include BLP goodness-of-fit rows when ``True``.

    Returns
    -------
    list[pathlib.Path]
        Paths of the household and diarrhea tables.
    """

    selected_results = results[
        results["specification"].isin(specifications)
    ].copy()

    table_definitions = [
        ("ecoli", ["SomeRiskHome", "VeryHighRiskHome"]),
        ("diarrhea", ["diarrhea"]),
    ]
    reported_effects = [
        ("IRM stacked", "Any Treatment"),
        ("APOS stacked", "Boiling"),
        ("APOS stacked", "Chlorination/tablets"),
        ("APOS stacked", "Straining/settling"),
    ]
    specification_labels = {
        "clustered_folds": "Clustered folds",
        "unclustered": "Ordinary folds",
    }

    output_paths = []
    for table_name, outcomes in table_definitions:
        table_results = selected_results[
            selected_results["outcome"].isin(outcomes)
        ]

        number_of_effect_columns = (
            len(specifications) * len(reported_effects)
        )
        # The first two columns identify the GATE group and show the underlying
        # source-water E. coli values. Treatment effects begin after them.
        total_columns = 2 + number_of_effect_columns
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, adjustbox}",
            r"\begin{landscape}",
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Heterogeneity of stacked GATE effects: {table_name}}}",
            rf"\label{{tab:{filename_prefix}-{table_name}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{ll{'c' * number_of_effect_columns}}}",
            r"\toprule",
        ]

        if len(specifications) == 1:
            effect_header = [
                "Heterogeneity group",
                r"\shortstack{Source E. coli range\\(CFU/100 mL)}",
            ]
            effect_header.extend(
                treatment for _, treatment in reported_effects
            )
            lines.append(" & ".join(effect_header) + r" \\")
        else:
            effect_header = [
                "Heterogeneity group",
                r"\shortstack{Source E. coli range\\(CFU/100 mL)}",
            ]
            for _, treatment in reported_effects:
                effect_header.append(
                    rf"\multicolumn{{{len(specifications)}}}{{c}}"
                    rf"{{{treatment}}}"
                )
            lines.append(" & ".join(effect_header) + r" \\")

            fold_header = ["", ""]
            for _ in reported_effects:
                for specification in specifications:
                    fold_header.append(
                        specification_labels[specification]
                    )
            lines.append(" & ".join(fold_header) + r" \\")

        lines.append(r"\midrule")

        for outcome_number, outcome in enumerate(outcomes):
            outcome_results = table_results[
                table_results["outcome"].eq(outcome)
            ]

            group_names = outcome_results["group"].drop_duplicates().tolist()
            for group_number, group_name in enumerate(group_names):
                group_results = outcome_results[
                    outcome_results["group"].eq(group_name)
                ]
                heterogeneity_label = group_results[
                    "heterogeneity_label"
                ].iloc[0]
                panel_label = (
                    f"{outcome_labels[outcome]}: {heterogeneity_label}"
                )
                if len(outcomes) > 1:
                    panel_letter = chr(65 + outcome_number * len(group_names) + group_number)
                    panel_label = f"Panel {panel_letter}: {panel_label}"
                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    rf"{{{panel_label}}} \\"
                )

                group_values = sorted(
                    group_results["group_value"].drop_duplicates(),
                    key=int,
                )
                for group_value in group_values:
                    one_group = group_results[
                        group_results["group_value"].eq(group_value)
                    ]

                    coefficient_cells = []
                    standard_error_cells = []
                    for method, treatment in reported_effects:
                        for specification in specifications:
                            selected = one_group[
                                one_group["method"].eq(method)
                                & one_group["treatment_label"].eq(treatment)
                                & one_group["specification"].eq(specification)
                            ]
                            result = selected.iloc[0]
                            standard_error = float(result["se"])
                            coefficient_cells.append(format_coefficient(
                                float(result["coef"]),
                                standard_error,
                                p_value=float(result["pval"]),
                            ))
                            standard_error_cells.append(
                                f"({standard_error:.3f})"
                            )

                    group_label = str(one_group["group_label"].iloc[0])
                    ecoli_range = str(
                        one_group["source_ecoli_range"].iloc[0]
                    )
                    lines.append(
                        " & ".join(
                            [group_label, ecoli_range] + coefficient_cells
                        )
                        + r" \\"
                    )
                    lines.append(
                        " & ".join(["", ""] + standard_error_cells)
                        + r" \\"
                    )

                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    r"{\textit{Sample}} \\"
                )
                for statistic in ("Observations", "PSUs"):
                    statistic_cells = [rf"\quad {statistic}", ""]
                    for method, treatment in reported_effects:
                        for specification in specifications:
                            method_results = group_results[
                                group_results["method"].eq(method)
                                & group_results["treatment_label"].eq(
                                    treatment
                                )
                                & group_results["specification"].eq(
                                    specification
                                )
                            ]

                            if statistic == "Observations":
                                statistic_value = (
                                    f"{int(method_results['sample_n'].iloc[0]):,}"
                                )
                            else:
                                if specification == "clustered_folds":
                                    statistic_value = (
                                        f"{int(method_results['sample_n_psu'].iloc[0]):,}"
                                    )
                                else:
                                    statistic_value = "---"
                            statistic_cells.append(statistic_value)
                    lines.append(" & ".join(statistic_cells) + r" \\")

                if include_blp_r2:
                    lines.append(
                        rf"\multicolumn{{{total_columns}}}{{l}}"
                        r"{\textit{BLP $R^{2}$}} \\"
                    )
                    for method, treatment in reported_effects:
                        r_squared_cells = [rf"\quad {treatment}", ""]
                        for candidate_method, candidate_treatment in reported_effects:
                            if (
                                candidate_method == method
                                and candidate_treatment == treatment
                            ):
                                for specification in specifications:
                                    selected = group_results[
                                        group_results["method"].eq(method)
                                        & group_results["treatment_label"].eq(
                                            treatment
                                        )
                                        & group_results["specification"].eq(
                                            specification
                                        )
                                    ]
                                    r_squared_cells.append(
                                        f"{float(selected['r2'].iloc[0]):.3f}"
                                    )
                            else:
                                r_squared_cells.extend(
                                    [""] * len(specifications)
                                )
                        lines.append(
                            " & ".join(r_squared_cells) + r" \\"
                        )

                if group_number < len(group_names) - 1:
                    lines.append(r"\addlinespace[4pt]")

            if outcome_number < len(outcomes) - 1:
                lines.append(r"\midrule")

        if tuple(specifications) == ("clustered_folds",):
            fold_note = (
                "Clustered folds use cluster-level sample splitting."
            )
        else:
            fold_note = (
                "Clustered folds use cluster-level sample splitting; "
                "ordinary folds use observation-level splitting."
            )
        r_squared_note = ""
        if include_blp_r2:
            r_squared_note = (
                " Weighted-score $R^2$ is descriptive; it is not a "
                "causal-model $R^2$."
            )
        gate_notes = (
            r"\begin{minipage}{\linewidth}\footnotesize \textit{Notes:} "
            + f"Coefficient rows report group-specific {estimand.upper()} estimates "
            + r"with significance stars; the following rows report standard errors. "
            + fold_note
            + r_squared_note
            + " Source-water E. coli ranges are measured in CFU/100 mL; "
            + "the upper group includes the top-coded value above 100."
            + " The heterogeneity analysis is exploratory."
            + r"\end{minipage}"
        )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            gate_notes,
            r"\end{table}",
            r"\end{landscape}",
        ])

        output_path = (
            Path(output_dir) / f"{filename_prefix}_{table_name}.tex"
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths

def write_gate_tables(
    gate_results, *, fold_mode="both", output_dir, estimand, outcome_labels,
):
    """Preserve combined GATE tables and publish each grouping separately."""
    output_paths = []
    required = {'source_ecoli', 'source_risk'}
    if set(gate_results['group']) != required:
        raise ValueError('Both source-water deciles and risk groups are required')
    for label, subset in [('', gate_results),
                          ('_deciles', gate_results.loc[gate_results.group.eq('source_ecoli')]),
                          ('_risk_groups', gate_results.loc[gate_results.group.eq('source_risk')])]:
        sections = [('main', ('unclustered',) if fold_mode == 'unclustered' else ('clustered_folds',), False)]
        if fold_mode == 'both':
            sections.append(('appendix', ('clustered_folds', 'unclustered'), True))
        for section, specifications, include_r2 in sections:
            paths = create_heterogeneity_comparison_tables(
                subset, output_dir=output_dir,
                filename_prefix=f'table_gate_{section}{label}',
                specifications=specifications, include_blp_r2=include_r2,
                estimand=estimand, outcome_labels=outcome_labels)
            output_paths.extend(paths)
            # Explicit captions distinguish the estimand, grouping and outcomes.
            grouping = {'': 'Deciles and Risk Groups', '_deciles': 'Source-Water Deciles',
                        '_risk_groups': 'Three Source-Water Risk Groups'}[label]
            for path in paths:
                text = path.read_text()
                for outcome, title in [('ecoli', 'Water Quality'), ('diarrhea', 'Diarrhea (U5)')]:
                    text = text.replace(
                        f'Heterogeneity of stacked GATE effects: {outcome}',
                        f"GATE ({estimand.upper()}) by {grouping}: {title}"
                        + (' -- Clustered and Unclustered' if section == 'appendix' else ''))
                path.write_text(text, encoding='utf-8')
    return output_paths
