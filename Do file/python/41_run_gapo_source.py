"""
GAPO by source-water E. coli risk — focused heterogeneity exhibit.

Step 4b of the heterogeneity workflow (a focused companion to 40_run_gate.py).
Where 40_run_gate.py sweeps every group in ``GATE_GROUPS``, this script reports
ONLY the source-water E. coli risk split (``RiskSource``: no risk / moderate /
very high), because that is the paper's substantive heterogeneity dimension
(point-of-use treatment is expected to matter little when source water is clean
and more when it is contaminated).

For each (dataset, outcome) a SINGLE DoubleMLAPOS is fit on the FULL sample
(multi-class propensity over WQ15_g, stacked learner, n_rep=1; identical to the
APOS headline, so RiskSource enters as a pre-treatment splitter rather than a
control).  It then reports, BY source-risk group:

  * GAPO  — native counterfactual LEVELS E[Y(d) | RiskSource] for every level
    (no treatment, boil, chlorine, filter, other).  "Predicted contamination /
    diarrhea under each method, within each source-risk environment."
    -> results_gapo_source.csv  and  Figures/gapo_source_<dataset>_<outcome>.png
  * GATE  — the ATE(d vs no treatment) contrast signal projected onto the risk
    groups (valid pointwise + JOINT/uniform bands) -> results_gate_source.csv.

Also writes a combined LaTeX table  Output/table_gapo_source.tex  (one panel per
outcome: methods x risk groups, GAPO level over (CI)).

Usage
-----
    cd "Do file/Python"
    python 41_run_gapo_source.py
    python 41_run_gapo_source.py --no-checkpoint   # refit APOS from scratch
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    OUTPUT_DIR, FIGURE_DIR, logger,
)
from _apos import APOS_LEVEL_LABELS
from _heterogeneity_apos import fit_base_apos, estimate_gate_apos, estimate_gapo_levels
from _runners import setup_environment, load_data

REFERENCE = 0  # no-treatment level
GROUP_VAR = "RiskSource"
GROUP_LABELS = {0: "No risk", 1: "Moderate", 2: "Very high"}
GROUP_ORDER = ["No risk", "Moderate", "Very high"]

HH_HET_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_HET_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]


# =============================================================================
# Estimation
# =============================================================================

def _run_dataset(dt, outcomes, dataset_type, confounders, skip_checkpoint=False):
    gapo_rows, gate_rows = [], []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"GAPO-source: outcome '{out_var}' missing in {dataset_type}, skipping")
            continue

        logger.info(f"\n=== GAPO/GATE by {GROUP_VAR} | {dataset_type} | {out['label']} ===")
        fitted = fit_base_apos(
            dt=dt, outcome_var=out_var, dataset_type=dataset_type,
            learner_name="stacked", confounder_groups=confounders, reference=REFERENCE,
            skip_checkpoint=skip_checkpoint,
        )
        if fitted is None:
            logger.warning(f"  base APOS failed for {out_var}")
            continue
        apos, dt_clean, levels = fitted

        if GROUP_VAR not in dt_clean.columns:
            logger.warning(f"  group '{GROUP_VAR}' missing in {dataset_type}, skipping")
            continue
        # RiskSource is stored as float (0.0/1.0/2.0); cast to int so the dummy
        # labels ("0"/"1"/"2") match the int-keyed GROUP_LABELS map.
        gseries = dt_clean[GROUP_VAR].reset_index(drop=True).astype(int)
        non_ref = [lv for lv in levels if lv != REFERENCE]

        # --- GAPO levels: E[Y(d) | RiskSource] for every level ---
        try:
            gapo = estimate_gapo_levels(apos, levels, gseries, group_labels=GROUP_LABELS)
            gapo.insert(0, "outcome", out_var)
            gapo.insert(1, "outcome_label", out["label"])
            gapo.insert(2, "dataset_type", dataset_type)
            gapo_rows.append(gapo)
            for _, r in gapo.iterrows():
                logger.info(
                    f"  GAPO[{r['treatment_label']:<11}|{str(r['group_label']):<10}] "
                    f"E[Y(d)]={r['gapo']:+.4f} [{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] (n={r['n']})"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  GAPO {GROUP_VAR} failed: {exc}")

        # --- GATE contrasts: ATE(d vs none) by RiskSource ---
        for lv in non_ref:
            trt_label = APOS_LEVEL_LABELS.get(lv, str(lv))
            try:
                gate_df = estimate_gate_apos(
                    apos, levels, lv, gseries, group_labels=GROUP_LABELS, reference=REFERENCE,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"    GATE {trt_label} failed: {exc}")
                continue
            gate_df.insert(0, "outcome", out_var)
            gate_df.insert(1, "outcome_label", out["label"])
            gate_df.insert(2, "dataset_type", dataset_type)
            gate_df.insert(3, "treatment_label", trt_label)
            gate_df.insert(4, "treatment_level", lv)
            gate_rows.append(gate_df)
            for _, r in gate_df.iterrows():
                logger.info(
                    f"  GATE[{trt_label:<11}|{str(r['group_label']):<10}] "
                    f"ATE={r['coef']:+.4f} [{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] "
                    f"joint[{r['ci_lower_joint']:+.3f},{r['ci_upper_joint']:+.3f}] (n={r['n']})"
                )
    return gapo_rows, gate_rows


# =============================================================================
# Figure: E[Y(d) | RiskSource] points with pointwise CIs, methods side-by-side
# =============================================================================

def _plot_gapo(gapo_df, out_var, out_label, dataset_type, out_dir=None):
    out_dir = out_dir or FIGURE_DIR
    sub = gapo_df[(gapo_df["outcome"] == out_var) & (gapo_df["dataset_type"] == dataset_type)]
    if sub.empty:
        return None

    levels = sorted(sub["treatment_level"].unique())
    groups = [g for g in GROUP_ORDER if g in set(sub["group_label"])]
    x = np.arange(len(groups))
    n_lv = len(levels)
    width = 0.8 / max(n_lv, 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, lv in enumerate(levels):
        rows = sub[sub["treatment_level"] == lv].set_index("group_label")
        y = [rows.loc[g, "gapo"] if g in rows.index else np.nan for g in groups]
        lo = [rows.loc[g, "gapo"] - rows.loc[g, "ci_lower"] if g in rows.index else np.nan for g in groups]
        hi = [rows.loc[g, "ci_upper"] - rows.loc[g, "gapo"] if g in rows.index else np.nan for g in groups]
        offset = (j - (n_lv - 1) / 2) * width
        ax.errorbar(
            x + offset, y, yerr=[lo, hi], fmt="o", capsize=3, markersize=5,
            label=APOS_LEVEL_LABELS.get(lv, str(lv)),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("Source-water E. coli risk")
    ax.set_ylabel(f"E[Y(d) | risk]  —  {out_label}")
    ax.set_title(f"GAPO by source risk: {out_label} ({dataset_type})")
    ax.margins(x=0.08)
    # legend OUTSIDE the panel, centered below, so it never overlaps the points
    ax.legend(frameon=False, ncol=n_lv, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              borderaxespad=0.0, columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout()

    png = out_dir / f"gapo_source_{dataset_type}_{out_var}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  figure -> {png.name}")
    return png


# =============================================================================
# LaTeX table: one panel per outcome, rows = methods, cols = risk groups
# =============================================================================

def _fmt_cell(level, lo, hi):
    # single-line cell: estimate + small CI (no in-cell \\, which would break the row)
    return f"{level:.3f}~{{\\scriptsize$[{lo:.3f},{hi:.3f}]$}}"


def create_gapo_source_table(gapo_df, filename="table_gapo_source.tex"):
    panels = []
    keys = gapo_df[["outcome", "outcome_label", "dataset_type"]].drop_duplicates().values.tolist()
    groups = GROUP_ORDER

    out = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]\\centering",
        "\\caption{Group average potential outcomes by source-water E. coli risk}",
        "\\label{tab:gapo_source}",
        "\\small\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{l" + "c" * len(groups) + "}",
        "\\toprule",
        "Method (potential outcome) & " + " & ".join(groups) + " \\\\",
        "\\midrule",
    ]

    for out_var, out_label, dtype in keys:
        sub = gapo_df[(gapo_df["outcome"] == out_var) & (gapo_df["dataset_type"] == dtype)]
        if sub.empty:
            continue
        out.append(f"\\multicolumn{{{len(groups)+1}}}{{l}}{{\\textit{{{out_label} ({dtype})}}}} \\\\")
        levels = sorted(sub["treatment_level"].unique())
        for lv in levels:
            rows = sub[sub["treatment_level"] == lv].set_index("group_label")
            cells = []
            for g in groups:
                if g in rows.index:
                    r = rows.loc[g]
                    cells.append(_fmt_cell(r["gapo"], r["ci_lower"], r["ci_upper"]))
                else:
                    cells.append("--")
            out.append(f"{APOS_LEVEL_LABELS.get(lv, str(lv))} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")

    out += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\vspace{3pt}",
        "\\begin{minipage}{\\linewidth}\\footnotesize \\textit{Notes:} Each cell is the "
        "estimated counterfactual mean outcome $E[Y(d)\\mid \\text{risk}]$ that would prevail "
        "if every household in the source-risk group used method $d$, from a full-sample "
        "\\texttt{DoubleMLAPOS} fit (stacked learner; same confounder vector as the headline "
        "APOS). 95\\% pointwise confidence intervals in brackets. These are potential-outcome "
        "\\emph{levels}; method-vs-none \\emph{effects} (with uniform bands) are in "
        "\\texttt{results\\_gate\\_source.csv}.\\end{minipage}",
        "\\end{table}",
    ]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    logger.info(f"LaTeX table -> {filepath.name}")
    return filepath


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="GAPO/GATE by source-water E. coli risk (RiskSource).")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: GAPO/GATE by source-water E. coli risk (RiskSource)")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    gapo_all, gate_all = [], []
    for dt, outs, dtype, conf in [
        (hh_dt, HH_HET_OUTCOMES, "HH", hh_conf),
        (u5_dt, U5_HET_OUTCOMES, "U5", u5_conf),
    ]:
        g, c = _run_dataset(dt, outs, dtype, conf, skip_checkpoint=args.no_checkpoint)
        gapo_all += g
        gate_all += c

    if gapo_all:
        gapo_df = pd.concat(gapo_all, ignore_index=True)
        gapo_df.to_csv(OUTPUT_DIR / "results_gapo_source.csv", index=False)
        gapo_df.to_pickle(OUTPUT_DIR / "results_gapo_source.pkl")
        logger.info("=" * 70)
        logger.info(f"GAPO levels -> {len(gapo_df)} rows -> results_gapo_source.csv")

        create_gapo_source_table(gapo_df)
        for out_var, out_label, dtype in (
            gapo_df[["outcome", "outcome_label", "dataset_type"]].drop_duplicates().values.tolist()
        ):
            _plot_gapo(gapo_df, out_var, out_label, dtype)
    else:
        logger.warning("GAPO-source produced no level results.")

    if gate_all:
        gate_df = pd.concat(gate_all, ignore_index=True)
        gate_df.to_csv(OUTPUT_DIR / "results_gate_source.csv", index=False)
        gate_df.to_pickle(OUTPUT_DIR / "results_gate_source.pkl")
        logger.info(f"GATE contrasts -> {len(gate_df)} rows -> results_gate_source.csv")
    else:
        logger.warning("GAPO-source produced no contrast results.")


if __name__ == "__main__":
    main()
