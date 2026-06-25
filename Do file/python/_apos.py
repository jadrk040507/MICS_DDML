"""
Multi-treatment estimation via DoubleMLAPOS (Average Potential Outcomes).

Section 10 of the analysis estimates each specific treatment (boil, chlorine,
filter, other) against "no treatment" using a *single binary IRM on a restricted
subsample*.  APOS is the joint, full-sample alternative: it estimates
E[Y(d)] for every treatment level d at once using AIPW (augmented IPW)
Neyman-orthogonal scores with a multi-class propensity P(D=d | X), then recovers
ATE(d vs 0) as the contrast E[Y(d)] - E[Y(0)].

Standard errors for the contrasts come from ``causal_contrast`` (correct, not the
conservative sqrt(V_d + V_0) approximation), with the same Chernozhukov et al.
(2022) sensitivity framework (RV / RVa) as the IRM path.

Note: ``DoubleMLAPOS`` does not support clustered standard errors in
doubleml 0.11.x, so APOS is estimated i.i.d.; this is recorded on each result
(``clustered=False``).  The clustered IRM results remain the headline estimates.

For U5 (child-level), an influence-function cluster-robust SE on HHID was
evaluated but discarded: households average only ~1.4 children (mostly
singletons), so the clustered SE moves <11% versus i.i.d. and changes no
conclusion — not worth the added complexity.
"""

import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import doubleml as dml
from scipy.stats import norm as _norm
from sklearn.base import clone
from _config import (
    N_FOLDS, N_REP, MIN_OBSERVATIONS, CHECKPOINT_DIR, logger,
)
from _data import create_model_matrix
from _learners import create_learners_for_binary
from _models import _extract_stacking_weights


def _avg_apo_weights(modellist, prefix: str):
    """Average stacked base-learner weights over per-level APO models.

    prefix='ml_g' aggregates the outcome-model keys (ml_g_d_lvl0/1); 'ml_m' the
    propensity. Returns {base: mean_weight} or None.
    """
    acc, cnt = {}, 0
    for apo in modellist:
        mods = getattr(apo, "models", None)
        if not mods:
            continue
        keys = [k for k in mods if k.startswith(prefix)]
        w = _extract_stacking_weights(mods, keys)
        if w:
            for base, val in w.items():
                acc[base] = acc.get(base, 0.0) + val
            cnt += 1
    if cnt == 0:
        return None
    return {base: round(v / cnt, 3) for base, v in acc.items()}

# WQ15_g -> integer APOS level (98 "other" recoded to 4 for contiguous coding)
APOS_LEVEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 98: 4}
APOS_LEVEL_LABELS = {0: "No Treatment", 1: "Boil", 2: "Chlorine", 3: "Filter", 4: "Other"}

SENS_CF = 0.03
SENS_RHO = 1.0

# Minimum observations per treatment level; sparser levels are dropped before
# fitting (a multi-class propensity cannot be cross-fit on a level that vanishes
# from a fold, and an APO for ~no observations is not estimable anyway).
APOS_MIN_LEVEL_N = 50


def _build_treat_cat(dt: pd.DataFrame) -> pd.Series:
    """Map WQ15_g to a contiguous integer treatment level (0..4)."""
    return dt["WQ15_g"].fillna(0).map(APOS_LEVEL_MAP).astype("Int64")


def estimate_apos(
    dt: pd.DataFrame,
    outcome_var: str,
    dataset_type: str,
    learner_name: str = "stacked",
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    reference: int = 0,
    n_folds: int = N_FOLDS,
    n_rep: int = N_REP,
    skip_checkpoint: bool = False,
) -> List[Dict[str, Any]]:
    """Fit APOS for one outcome+learner and return ATE(d vs reference) rows.

    ``ml_g`` is the binary-outcome regressor and ``ml_m`` the (multi-class)
    propensity classifier for ``learner_name`` from ``create_learners_for_binary``
    (LogisticRegressionCV / RF / XGB / StackingClassifier all support multi-class
    ``predict_proba`` over the treatment levels).

    Successful result lists are cached to ``Output/Checkpoints`` keyed by
    ``(dataset, outcome, learner)``; delete those files to force a refit (e.g.
    after changing the confounders, folds, or learner definitions).

    Returns one result dict per non-reference treatment level, using the same
    key schema as ``models.estimate_effect`` so the table writers can consume it.
    """
    cp_file = CHECKPOINT_DIR / f"apos_{dataset_type}_{outcome_var}_{learner_name}.pkl"
    if not skip_checkpoint and cp_file.is_file():
        try:
            with open(cp_file, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"    Loaded APOS checkpoint: {cp_file.name}")
            return cached
        except Exception as e:  # noqa: BLE001
            logger.warning(f"    Could not load APOS checkpoint {cp_file.name}: {e}")

    dt = dt.copy()
    dt["treat_cat"] = _build_treat_cat(dt)

    X, dt = create_model_matrix(dt, outcome_var, confounder_groups=confounder_groups)

    mask = (
        dt[outcome_var].notna()
        & dt["treat_cat"].notna()
        & ~np.isnan(X).any(axis=1)
        & ~np.isinf(X).any(axis=1)
    )
    n = int(mask.sum())
    if n < MIN_OBSERVATIONS:
        logger.error(f"    APOS {outcome_var}: too few complete obs (N={n})")
        return []

    X = X[mask.values]
    y = dt.loc[mask, outcome_var].to_numpy(dtype=float)
    d = dt.loc[mask, "treat_cat"].astype(int).to_numpy()

    # Drop sparse treatment levels (keep the reference regardless)
    counts = pd.Series(d).value_counts()
    levels = sorted(
        int(lv) for lv in counts.index
        if counts[lv] >= APOS_MIN_LEVEL_N or lv == reference
    )
    dropped = [int(lv) for lv in counts.index if int(lv) not in levels]
    if dropped:
        logger.warning(
            f"    APOS {outcome_var}: dropping sparse levels "
            f"{[APOS_LEVEL_LABELS.get(lv, lv) for lv in dropped]} "
            f"(< {APOS_MIN_LEVEL_N} obs)"
        )
        keep_rows = np.isin(d, levels)
        X, y, d = X[keep_rows], y[keep_rows], d[keep_rows]
        n = int(keep_rows.sum())
    if reference not in levels or len(levels) < 2:
        logger.error(f"    APOS {outcome_var}: insufficient treatment levels {levels}")
        return []

    lset = create_learners_for_binary()[learner_name]
    ml_g = lset["g"]            # outcome regressor (ProbaRegressor for binary Y)
    ml_m = lset["m"]            # multi-class propensity classifier P(D=d | X)

    data = dml.DoubleMLData.from_arrays(x=X, y=y, d=d)
    apos = dml.DoubleMLAPOS(
        obj_dml_data=data,
        ml_g=clone(ml_g), ml_m=clone(ml_m),
        treatment_levels=levels,
        n_folds=n_folds, n_rep=n_rep,
        trimming_rule="truncate", trimming_threshold=0.01,
    )
    logger.info(f"    APOS {outcome_var} [{learner_name}]: fitting (levels={levels}, N={n:,}) ...")
    store_models = (learner_name == "stacked")
    apos.fit(store_models=store_models)

    # Stacking weights (averaged over per-level APO models and folds)
    weights_g = weights_m = None
    if store_models and getattr(apos, "modellist", None):
        weights_g = _avg_apo_weights(apos.modellist, prefix="ml_g")
        weights_m = _avg_apo_weights(apos.modellist, prefix="ml_m")

    contrast = apos.causal_contrast(reference_levels=reference)
    csum = contrast.summary
    coefs = csum["coef"].to_numpy()
    ses = csum["std err"].to_numpy()
    ci_lo = csum["2.5 %"].to_numpy()
    ci_hi = csum["97.5 %"].to_numpy()

    rv = rva = None
    try:
        contrast.sensitivity_analysis(cf_y=SENS_CF, cf_d=SENS_CF, rho=SENS_RHO)
        sp = contrast.sensitivity_params
        if sp is not None:
            rv = np.ravel(sp["rv"]).astype(float)
            rva = np.ravel(sp["rva"]).astype(float)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"    APOS {outcome_var}: sensitivity failed: {exc}")

    non_ref = [lv for lv in levels if lv != reference]
    results = []
    for i, lv in enumerate(non_ref):
        results.append({
            "outcome": outcome_var,
            "treatment": APOS_LEVEL_LABELS.get(lv, str(lv)),
            "treatment_level": lv,
            "learner": learner_name,
            "method": "APOS",
            "dataset_type": dataset_type,
            "clustered": False,
            "coef": float(coefs[i]),
            "se": float(ses[i]),
            "ci_lower": float(ci_lo[i]),
            "ci_upper": float(ci_hi[i]),
            "pval": float(2 * _norm.sf(abs(coefs[i] / ses[i]))) if ses[i] else np.nan,
            "rv_q": float(rv[i]) if rv is not None and i < len(rv) else None,
            "rv_qa": float(rva[i]) if rva is not None and i < len(rva) else None,
            "n": n,
            "n_treated": int((d == lv).sum()),
            "y_mean": float(np.mean(y)),
            "weights_g": weights_g,
            "weights_m": weights_m,
        })
        logger.info(
            f"      {APOS_LEVEL_LABELS.get(lv, lv):<10} ATE={coefs[i]:+.4f} ({ses[i]:.4f})"
        )

    if not skip_checkpoint and results:
        try:
            with open(cp_file, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"    Saved APOS checkpoint: {cp_file.name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"    Could not save APOS checkpoint {cp_file.name}: {e}")

    return results


def run_apos(
    dt: pd.DataFrame,
    outcomes: List[Dict[str, str]],
    dataset_type: str,
    learner_names: Optional[List[str]] = None,
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    n_folds: int = N_FOLDS,
    n_rep: int = N_REP,
    skip_checkpoint: bool = False,
) -> List[Dict[str, Any]]:
    """Run APOS over every outcome x learner in a dataset (checkpointed)."""
    from _config import LEARNER_NAMES
    learner_names = learner_names or LEARNER_NAMES
    results = []
    for out in outcomes:
        out_var = out["var"]
        if out_var not in dt.columns:
            logger.warning(f"APOS: outcome '{out_var}' not found, skipping")
            continue
        for ln in learner_names:
            logger.info(f"APOS | {out.get('label', out_var)} ({dataset_type}) | {ln}")
            try:
                results += estimate_apos(
                    dt=dt, outcome_var=out_var, dataset_type=dataset_type,
                    learner_name=ln, confounder_groups=confounder_groups,
                    n_folds=n_folds, n_rep=n_rep, skip_checkpoint=skip_checkpoint,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"    APOS {out_var} [{ln}] failed: {exc}")
    return results

