"""
Rob_1 overlap robustness check.

Constructs a sample-restriction flag ``Rob_1`` that drops households whose
treatment method has *weak within-country support* --- i.e. a (country, method)
cell where the method's within-country share is below 5 percent or fewer than 10
households use it.  Only TREATED levels are flagged (the no-treatment baseline is
never dropped for being dominant); this targets the orphan-treated, no-overlap
observations (notably chlorine, which is concentrated in a few countries).

We then re-run BOTH estimators on the restricted sample (Rob_1 == 0), with
separate output names, and compare to the headline:

  * if the IRM/APOS divergence (e.g. the chlorine sign-flip) DISAPPEARS once the
    weak-overlap cells are dropped, the divergence was driven by those cells ->
    overlap is the culprit and APOS is justified;
  * if it PERSISTS, overlap is not the story.

Headline results are untouched; this writes results_rob1_irm.csv /
results_rob1_apos.csv and prints a headline-vs-Rob_1 comparison.  Default learner
``stacked`` (the headline); APOS re-runs skip checkpoints to avoid colliding with
the full-sample caches, IRM re-runs use ``rob1_*`` checkpoint prefixes.
"""

import argparse

import numpy as np
import pandas as pd

from _config import (
    BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS,
    ANY_TREATMENT, SPECIFIC_TREATMENTS, OUTPUT_DIR, N_FOLDS, logger,
)
from _models import run_analysis
from _apos import run_apos, _build_treat_cat
from _runners import setup_environment, load_data, select_learners

SHARE_LO = 0.05      # drop a (country, method) cell below this within-country share
MIN_COUNT = 10       # ... or with fewer than this many users
COUNTRY_COL = "country_cat"

HH_OUT = [{"var": "SomeRiskHome", "label": "Some Risk Home"},
          {"var": "VeryHighRiskHome", "label": "Very High Risk Home"}]
U5_OUT = [{"var": "diarrhea", "label": "Diarrhea"}]


def add_rob1(dt):
    """Add ``Rob_1`` = 1 for treated households in weak-support country cells."""
    d = dt.copy()
    d["_lvl"] = _build_treat_cat(d)
    valid = d[COUNTRY_COL].notna() & d["_lvl"].notna()
    cnt = d[valid].groupby([COUNTRY_COL, "_lvl"]).size().rename("cnt").reset_index()
    tot = d[valid].groupby(COUNTRY_COL).size().rename("ctot").reset_index()
    cell = cnt.merge(tot, on=COUNTRY_COL)
    cell["share"] = cell["cnt"] / cell["ctot"]
    # weak only for treated levels (lvl != 0); baseline never dropped for dominance
    cell["weak"] = (((cell["share"] < SHARE_LO) | (cell["cnt"] < MIN_COUNT))
                    & (cell["_lvl"] != 0)).astype(int)
    d = d.merge(cell[[COUNTRY_COL, "_lvl", "weak"]], on=[COUNTRY_COL, "_lvl"], how="left")
    d["Rob_1"] = d["weak"].fillna(0).astype(int)
    return d.drop(columns=["_lvl", "weak"])


def _irm_rows(results, method="IRM"):
    out = []
    for r in results:
        if r is None:
            continue
        out.append({
            "method": method, "dataset_type": r.get("dataset_type"),
            "outcome": r.get("outcome"), "treatment": r.get("treatment"),
            "learner": r.get("learner"), "n": r.get("n"),
            "coef": r.get("coef"), "se": r.get("se"),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Rob_1 overlap robustness (restricted-sample IRM + APOS).")
    ap.add_argument("--learners", nargs="+", default=["stacked"])
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-rep", type=int, default=1)
    args = ap.parse_args()

    setup_environment()
    logger.info("=" * 70)
    logger.info(f"MICS DDML: Rob_1 OVERLAP ROBUSTNESS (share<{SHARE_LO}, count<{MIN_COUNT}; "
                f"learners={args.learners})")
    logger.info("=" * 70)

    hh_dt, u5_dt = load_data()
    hh_dt, u5_dt = add_rob1(hh_dt), add_rob1(u5_dt)
    hh_r = hh_dt[hh_dt["Rob_1"] == 0].copy()
    u5_r = u5_dt[u5_dt["Rob_1"] == 0].copy()
    logger.info(f"Rob_1: HH keep {len(hh_r):,}/{len(hh_dt):,} "
                f"(drop {len(hh_dt)-len(hh_r):,}); U5 keep {len(u5_r):,}/{len(u5_dt):,} "
                f"(drop {len(u5_dt)-len(u5_r):,})")

    learners = select_learners(names=args.learners)
    hh_conf = dict(BASE_CONFOUNDERS)
    u5_conf = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    # ---- IRM on the restricted sample (any + specific vs none) ----
    irm = []
    irm += run_analysis(hh_r, HH_OUT, [ANY_TREATMENT], learners, hh_conf,
                        dataset_type="HH", prefix="rob1_hh_any",
                        n_folds=args.n_folds, n_rep=args.n_rep)
    irm += run_analysis(hh_r, HH_OUT, SPECIFIC_TREATMENTS, learners, hh_conf,
                        dataset_type="HH", prefix="rob1_hh_specific_none", vs_none=True,
                        n_folds=args.n_folds, n_rep=args.n_rep)
    irm += run_analysis(u5_r, U5_OUT, [ANY_TREATMENT], learners, u5_conf,
                        dataset_type="U5", prefix="rob1_u5_any",
                        n_folds=args.n_folds, n_rep=args.n_rep)
    irm += run_analysis(u5_r, U5_OUT, SPECIFIC_TREATMENTS, learners, u5_conf,
                        dataset_type="U5", prefix="rob1_u5_specific_none", vs_none=True,
                        n_folds=args.n_folds, n_rep=args.n_rep)

    # ---- APOS on the restricted sample (skip checkpoints: distinct sample) ----
    apos = []
    apos += run_apos(hh_r, HH_OUT, "HH", learner_names=args.learners,
                     confounder_groups=hh_conf, n_folds=args.n_folds,
                     n_rep=args.n_rep, skip_checkpoint=True)
    apos += run_apos(u5_r, U5_OUT, "U5", learner_names=args.learners,
                     confounder_groups=u5_conf, n_folds=args.n_folds,
                     n_rep=args.n_rep, skip_checkpoint=True)

    irm_df = pd.DataFrame(_irm_rows(irm))
    apos_df = pd.DataFrame(_irm_rows(apos, method="APOS"))
    irm_df.to_csv(OUTPUT_DIR / "results_rob1_irm.csv", index=False)
    apos_df.to_csv(OUTPUT_DIR / "results_rob1_apos.csv", index=False)
    logger.info(f"Rob_1 IRM  -> {len(irm_df)} rows -> results_rob1_irm.csv")
    logger.info(f"Rob_1 APOS -> {len(apos_df)} rows -> results_rob1_apos.csv")

    # ---- headline vs Rob_1 comparison (focus: does chlorine still flip?) ----
    logger.info("=" * 70)
    logger.info("HEADLINE vs Rob_1 (stacked)")
    logger.info("=" * 70)
    try:
        h_irm = pd.read_csv(OUTPUT_DIR / "results_main.csv")
        h_apos = pd.read_csv(OUTPUT_DIR / "results_apos.csv")
        for label, hdf, rdf in [("IRM", h_irm, irm_df), ("APOS", h_apos, apos_df)]:
            hh1 = hdf[(hdf.get("learner") == "stacked")] if "learner" in hdf else hdf
            for _, rr in rdf[rdf["learner"] == "stacked"].iterrows():
                key = (rr["outcome"], rr["treatment"], rr["dataset_type"])
                hm = hh1[(hh1["outcome"] == rr["outcome"]) &
                         (hh1["treatment"].astype(str) == str(rr["treatment"]))]
                hc = float(hm["coef"].iloc[0]) if len(hm) else float("nan")
                logger.info(f"  {label:<5} {str(rr['treatment']):<16} {rr['outcome']:<18} "
                            f"headline={hc:+.4f}  Rob_1={float(rr['coef']):+.4f}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  comparison skipped: {exc}")


if __name__ == "__main__":
    main()
