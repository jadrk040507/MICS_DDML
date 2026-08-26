"""GATE — Group Average Treatment Effects (APOS, projection-based).

For each (dataset, outcome) a SINGLE DoubleMLAPOS is fit on the FULL sample
(multi-class propensity over WQ15_g levels, stacked learner, n_rep=1).  Then, for
each specific method d (boil / chlorine / filter / other) vs no treatment, and
each grouping variable in ``GATE_GROUPS``:

the ATE(d vs 0) contrast signal is projected onto group indicators
(``DoubleMLBLP``, is_gate=True) -> per-group effect with pointwise + JOINT
(uniform, Semenova-Chernozhukov) bands. Written to ``results_gate.csv``.

The full confounder vector (identical to the APOS headline, incl. RiskSource via
the source-E.coli deciles) enters the propensity / outcome models, so splitting
GATE by RiskSource is the econometrically correct conditional contrast
(RiskSource is pre-treatment).  APOS replaces the old IRM single-method-subsample
path (positivity failure, cf_d ~ 1).
"""

import argparse

import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    GATE_GROUPS, HELEVEL_CODES, TOILET_CODES, WS1G_CODES,
    OUTPUT_DIR, logger,
)
from _apos import APOS_LEVEL_LABELS
from _heterogeneity_apos import fit_base_apos, estimate_gate_apos
from _runners import setup_environment, load_data

REFERENCE = 0  # no-treatment level

HH_HET_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_HET_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]

GROUP_LABEL_MAPS = {
    "RiskSource": {0: "No risk", 1: "Moderate", 2: "Very high"},
    "helevel": HELEVEL_CODES,
    "windex5": {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"},
    "country_cat": None,
    "Toilet": TOILET_CODES,
    "WS1_g": WS1G_CODES,
    "num_children_bin": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"},
    "child_age": None,
}
RISKSOURCE_ORDER = ["No risk", "Moderate", "Very high"]


def _normalize_key(value):
    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except Exception:  # noqa: BLE001
        return value


def _sort_key(value):
    norm = _normalize_key(value)
    if isinstance(norm, (int, float)):
        return (0, norm)
    return (1, str(norm))


def _stars(pval):
    if pd.isna(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.1:
        return "*"
    return ""


def _fmt_cell(coef, se, pval):
    return f"{coef:.3f}{_stars(pval)} ({se:.3f})"


def _table_env(n_groups):
    return "sidewaystable" if n_groups > 8 else "table"


def _table_name(group_var):
    return group_var.lower().replace("_", "")


def _panel_label(index):
    return f"Panel {chr(ord('A') + index)}"


def _group_order(group_var, sub_all):
    labels = list(sub_all["group_label"].drop_duplicates())
    if group_var == "RiskSource":
        return RISKSOURCE_ORDER
    if group_var == "windex5":
        return ["Q1", "Q2", "Q3", "Q4", "Q5"]
    if group_var == "helevel":
        preferred = ["none", "primary", "secondary", "missing"]
        return [lab for lab in preferred if lab in labels] + [lab for lab in labels if lab not in preferred]
    if group_var in {"num_children_bin", "child_age"}:
        return [str(v) for v in sorted({_normalize_key(v) for v in labels}, key=_sort_key)]
    return [str(v) for v in sorted(labels, key=_sort_key)]


def _lookup_group_row(sub, group_label):
    hit = sub[sub["group_label"] == group_label]
    if hit.empty:
        hit = sub[sub["group_value"].astype(str) == str(group_label)]
    return hit


def _write_gate_table(group_var, sub_all, filename):
    if sub_all.empty:
        logger.warning(f"No {group_var} GATE rows; table skipped.")
        return None

    keys = sub_all[["outcome", "dataset_type"]].drop_duplicates().values.tolist()
    group_order = _group_order(group_var, sub_all)
    ncols = len(group_order) + 1
    env = _table_env(len(group_order))
    caption = {
        "RiskSource": "Group average treatment effects by source-water E. coli risk",
        "helevel": "Group average treatment effects by household education",
        "windex5": "Group average treatment effects by wealth quintile",
        "country_cat": "Group average treatment effects by country",
        "Toilet": "Group average treatment effects by toilet type",
        "WS1_g": "Group average treatment effects by water source type",
        "num_children_bin": "Group average treatment effects by number of children",
        "child_age": "Group average treatment effects by child age",
    }.get(group_var, f"Group average treatment effects by {group_var}")

    out = [
        "% Requires: \\usepackage{booktabs}",
        f"\\begin{{{env}}}[p]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:gate_{_table_name(group_var)}}}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + ("c" * len(group_order)) + "}",
        "\\toprule",
    ]

    for idx, (out_var, dtype) in enumerate(keys):
        sub = sub_all[(sub_all["outcome"] == out_var) & (sub_all["dataset_type"] == dtype)].copy()
        if sub.empty:
            continue
        out.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{{_panel_label(idx)}: {out_var} ({dtype})}}}}\\\\")
        out.append("Method vs. no treatment & " + " & ".join(group_order) + " \\\\")
        out.append("\\midrule")
        treatment_col = "treatment_level" if "treatment_level" in sub.columns else "treatment_label"
        for lv in sorted(sub[treatment_col].dropna().unique()):
            rows = sub[sub[treatment_col] == lv]
            label = rows["treatment_label"].iloc[0]
            cells = []
            for group in group_order:
                hit = _lookup_group_row(rows, group)
                if hit.empty:
                    cells.append("--")
                else:
                    r = hit.iloc[0]
                    cells.append(_fmt_cell(r["coef"], r["se"], r["pval"]))
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if idx != len(keys) - 1:
            out.append("\\addlinespace")

    out += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{flushleft}",
        "\\footnotesize",
        "\\textit{Notes:} Cells report APOS-based GATE estimates. For each treatment, we first construct the APOS contrast signal versus no treatment and then project that signal onto the group indicators to obtain the group-specific effect. Standard errors are in parentheses. Stars denote p-values: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        "\\end{flushleft}",
        f"\\end{{{env}}}",
    ]
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    logger.info(f"LaTeX table -> {filepath.name}")
    return filepath


def _run_dataset(dt, outcomes, dataset_type, confounders, skip_checkpoint=False):
    gate_rows = []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"GATE: outcome '{out_var}' missing in {dataset_type}, skipping")
            continue

        logger.info(f"\n=== GATE | {dataset_type} | {out['label']} ===")
        fitted = fit_base_apos(
            dt=dt, outcome_var=out_var, dataset_type=dataset_type,
            learner_name="stacked", confounder_groups=confounders, reference=REFERENCE,
            skip_checkpoint=skip_checkpoint,
        )
        if fitted is None:
            logger.warning(f"  base APOS failed for {out_var}")
            continue
        apos, dt_clean, levels = fitted
        non_ref = [lv for lv in levels if lv != REFERENCE]

        for grp in GATE_GROUPS:
            if dataset_type not in grp["datasets"]:
                continue
            gvar = grp["var"]
            if gvar not in dt_clean.columns:
                logger.warning(f"  group '{gvar}' missing, skipping")
                continue
            gseries = dt_clean[gvar].reset_index(drop=True)
            glabels = GROUP_LABEL_MAPS.get(gvar)

            for lv in non_ref:
                trt_label = APOS_LEVEL_LABELS.get(lv, str(lv))
                logger.info(f"  GATE {trt_label} vs none over {grp['label']} ({gvar})")
                try:
                    gate_df = estimate_gate_apos(
                        apos, levels, lv, gseries, group_labels=glabels, reference=REFERENCE,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"    GATE {gvar} [{trt_label}] failed: {exc}")
                    continue
                gate_df.insert(0, "outcome", out_var)
                gate_df.insert(1, "dataset_type", dataset_type)
                gate_df.insert(2, "treatment", trt_label)
                gate_df.insert(3, "treatment_label", trt_label)
                gate_df.insert(4, "treatment_level", lv)
                gate_df.insert(5, "group_var", gvar)
                gate_df.insert(6, "group_var_label", grp["label"])
                for _, r in gate_df.iterrows():
                    logger.info(
                        f"    {str(r['group_label']):<12} GATE={r['coef']:+.4f} "
                        f"[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] "
                        f"joint[{r['ci_lower_joint']:+.3f},{r['ci_upper_joint']:+.3f}] (n={r['n']})"
                    )
                gate_rows.append(gate_df)
    return gate_rows


def create_gate_source_table(gate_df, filename="table_gate_risksource.tex"):
    return _write_gate_table("RiskSource", gate_df[gate_df["group_var"] == "RiskSource"].copy(), filename)


def create_gate_tables(gate_df):
    files = []
    for group_var in gate_df["group_var"].drop_duplicates():
        file = _write_gate_table(
            group_var,
            gate_df[gate_df["group_var"] == group_var].copy(),
            f"table_gate_{_table_name(group_var)}.tex",
        )
        if file is not None:
            files.append(file)
    return files


def main():
    ap = argparse.ArgumentParser(description="APOS GATE group effects.")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: GATE (APOS projection-based group effects)")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    gate_all = []
    for dt, outs, dtype, conf in [
        (hh_dt, HH_HET_OUTCOMES, "HH", hh_conf),
        (u5_dt, U5_HET_OUTCOMES, "U5", u5_conf),
    ]:
        gate_all += _run_dataset(dt, outs, dtype, conf, skip_checkpoint=args.no_checkpoint)

    if gate_all:
        gate_df = pd.concat(gate_all, ignore_index=True)
        gate_df.to_csv(OUTPUT_DIR / "results_gate.csv", index=False)
        gate_df.to_pickle(OUTPUT_DIR / "results_gate.pkl")
        logger.info("=" * 70)
        logger.info(f"GATE complete — {len(gate_df)} group effects -> results_gate.csv")
        create_gate_tables(gate_df)
    else:
        logger.warning("GATE produced no results.")

if __name__ == "__main__":
    main()
