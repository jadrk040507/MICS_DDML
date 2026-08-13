"""
Step 2 of the workflow: Multi-treatment APOS analysis.

Estimates the Average Potential Outcome contrasts ATE(d vs no treatment) for each
specific water-treatment method (boil, chlorine, filter, other) jointly on the
full sample, for every outcome.  Complements the binary subsample IRM in
``10_run_irm.py`` (specific treatments) with the efficient full-sample AIPW
estimator.

    cd "Do file/python"
    python 20_run_apos.py
"""

import logging
import pickle
from _config import BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS, OUTPUT_DIR
from _apos import run_apos
from _runners import setup_environment, load_data, save_results

# E.coli outcomes -> HH; diarrhea -> U5.  The headline results are split into a
# standalone E.coli table and a standalone diarrhea table (each with the
# any-treatment IRM as the top row over the APOS contrasts).
HH_APOS_OUTCOMES = [
    {"var": "SomeRiskHome", "label": "Some Risk Home"},
    {"var": "VeryHighRiskHome", "label": "Very High Risk Home"},
]
U5_APOS_OUTCOMES = [{"var": "diarrhea", "label": "Diarrhea"}]

ECOLI_OUTCOMES = ["SomeRiskHome", "VeryHighRiskHome"]
DIARRHEA_OUTCOMES = ["diarrhea"]


def _load_irm_any():
    """Load the any-treatment IRM results saved by 10_run_irm.py.

    Returns the any-treatment IRM rows so they can sit on top of the APOS panel.
    Empty (with a warning) if 10_run_irm.py has not been run yet.
    """
    path = OUTPUT_DIR / "results_main.pkl"
    if not path.exists():
        logging.getLogger("mics_ddml").warning(
            "    results_main.pkl not found; run 10_run_irm.py first. "
            "Results tables will omit the any-treatment IRM row.")
        return []
    with open(path, "rb") as f:
        res = pickle.load(f)
    return [r for r in res
            if r.get("method") != "APOS"
            and r.get("treatment") == "water_treatment"
            and r.get("subgroup_var") is None]


def main():
    setup_environment()
    log = logging.getLogger("mics_ddml")
    log.info("=" * 70)
    log.info("MICS DDML: APOS MULTI-TREATMENT ANALYSIS (all learners)")
    log.info("=" * 70)

    hh_dt, u5_dt = load_data()

    hh_confounders = BASE_CONFOUNDERS.copy()
    u5_confounders = {**BASE_CONFOUNDERS, **U5_ADDITIONAL_CONFOUNDERS}

    all_results = []
    all_results += run_apos(hh_dt, HH_APOS_OUTCOMES, dataset_type="HH",
                            confounder_groups=hh_confounders)
    all_results += run_apos(u5_dt, U5_APOS_OUTCOMES, dataset_type="U5",
                            confounder_groups=u5_confounders)

    save_results(all_results, tag="apos")

    try:
        from _tables import create_results_table
        # Any-treatment IRM (from 10_run_irm.py) sits on top of the APOS panel.
        combined = _load_irm_any() + all_results
        # Section 2: E.coli contamination — 2 outcomes x 7 learners = 15 cols,
        # too wide for portrait, so landscape.
        create_results_table(
            combined, outcomes=ECOLI_OUTCOMES,
            filename="table_results_ecoli.tex",
            caption=r"DDML Estimates: Water Treatment on E.coli Contamination",
            label="tab:results-ecoli", landscape=True)
        # Section 3: Diarrhea (standalone) — single outcome (8 cols) reads better
        # upright, so portrait.
        create_results_table(
            combined, outcomes=DIARRHEA_OUTCOMES,
            filename="table_results_diarrhea.tex",
            caption=r"DDML Estimates: Water Treatment on Child Diarrhea (under-5)",
            label="tab:results-diarrhea", landscape=False)
    except Exception as e:
        logging.getLogger("mics_ddml").warning(f"    Table generation failed: {e}")

    log.info("=" * 70)
    log.info("APOS ANALYSIS COMPLETE")
    log.info(f"Results saved to: {OUTPUT_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

