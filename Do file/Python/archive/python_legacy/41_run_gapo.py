"""GAPO — Group Average Potential Outcomes (APOS levels).

For each dataset/outcome, fit one DoubleMLAPOS on full sample. For each group
in GATE_GROUPS, report native potential-outcome levels E[Y(d) | group] for all
treatment levels, including no treatment. Written to results_gapo.csv.
"""

import argparse

import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    GATE_GROUPS, HELEVEL_CODES, TOILET_CODES, WS1G_CODES,
    OUTPUT_DIR, logger,
)
from _heterogeneity_apos import fit_base_apos, estimate_gapo_levels
from _runners import setup_environment, load_data

REFERENCE = 0

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


def _fmt_cell(value, lo, hi):
    return f"{value:.3f}~{{\\scriptsize$[{lo:.3f},{hi:.3f}]$}}"


def _run_dataset(dt, outcomes, dataset_type, confounders, skip_checkpoint=False):
    gapo_rows = []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"GAPO: outcome '{out_var}' missing in {dataset_type}, skipping")
            continue

        logger.info(f"\n=== GAPO | {dataset_type} | {out['label']} ===")
        fitted = fit_base_apos(
            dt=dt, outcome_var=out_var, dataset_type=dataset_type,
            learner_name="stacked", confounder_groups=confounders, reference=REFERENCE,
            skip_checkpoint=skip_checkpoint,
        )
        if fitted is None:
            logger.warning(f"  base APOS failed for {out_var}")
            continue
        apos, dt_clean, levels = fitted

        for grp in GATE_GROUPS:
            if dataset_type not in grp["datasets"]:
                continue
            gvar = grp["var"]
            if gvar not in dt_clean.columns:
                logger.warning(f"  group '{gvar}' missing, skipping")
                continue
            logger.info(f"  GAPO levels over {grp['label']} ({gvar})")
            try:
                gapo = estimate_gapo_levels(
                    apos, levels, dt_clean[gvar].reset_index(drop=True),
                    group_labels=GROUP_LABEL_MAPS.get(gvar),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"    GAPO {gvar} failed: {exc}")
                continue
            gapo.insert(0, "outcome", out_var)
            gapo.insert(1, "dataset_type", dataset_type)
            gapo.insert(2, "group_var", gvar)
            gapo.insert(3, "group_var_label", grp["label"])
            gapo_rows.append(gapo)
    return gapo_rows


def create_gapo_source_table(gapo_df, filename="table_gapo_source.tex"):
    """Paper table: RiskSource GAPO levels only."""
    sub_all = gapo_df[gapo_df["group_var"] == "RiskSource"].copy()
    if sub_all.empty:
        logger.warning("No RiskSource GAPO rows; table skipped.")
        return None

    keys = sub_all[["outcome", "dataset_type"]].drop_duplicates().values.tolist()
    out = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]\\centering",
        "\\caption{Group average potential outcomes by source-water E. coli risk}",
        "\\label{tab:gapo_source}",
        "\\small\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method (potential outcome) & No risk & Moderate & Very high \\\\",
        "\\midrule",
    ]
    for out_var, dtype in keys:
        sub = sub_all[(sub_all["outcome"] == out_var) & (sub_all["dataset_type"] == dtype)]
        out.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{out_var} ({dtype})}}}} \\\\")
        treatment_col = "treatment_level" if "treatment_level" in sub.columns else "treatment_label"
        for lv in sorted(sub[treatment_col].unique()):
            rows = sub[sub[treatment_col] == lv].set_index("group_label")
            label = rows["treatment_label"].iloc[0]
            cells = []
            for group in RISKSOURCE_ORDER:
                if group in rows.index:
                    r = rows.loc[group]
                    cells.append(_fmt_cell(r["gapo"], r["ci_lower"], r["ci_upper"]))
                else:
                    cells.append("--")
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")
    out += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    logger.info(f"LaTeX table -> {filepath.name}")
    return filepath


def main():
    ap = argparse.ArgumentParser(description="APOS GAPO group potential-outcome levels.")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="ignore cached APOS fits and refit from scratch")
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info("MICS DDML: GAPO (APOS group potential-outcome levels)")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    gapo_all = []
    for dt, outs, dtype, conf in [
        (hh_dt, HH_HET_OUTCOMES, "HH", hh_conf),
        (u5_dt, U5_HET_OUTCOMES, "U5", u5_conf),
    ]:
        gapo_all += _run_dataset(dt, outs, dtype, conf, skip_checkpoint=args.no_checkpoint)

    if gapo_all:
        gapo_df = pd.concat(gapo_all, ignore_index=True)
        gapo_df.to_csv(OUTPUT_DIR / "results_gapo.csv", index=False)
        gapo_df.to_pickle(OUTPUT_DIR / "results_gapo.pkl")
        logger.info("=" * 70)
        logger.info(f"GAPO complete — {len(gapo_df)} group rows -> results_gapo.csv")
        create_gapo_source_table(gapo_df)
    else:
        logger.warning("GAPO produced no results.")


if __name__ == "__main__":
    main()
