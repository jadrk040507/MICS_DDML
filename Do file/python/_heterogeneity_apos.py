"""
APOS-based heterogeneity: potential-outcome levels (GAPO/CAPO) and treatment
effects (GATE/CATE) for each specific method vs. no treatment.

Why APOS instead of IRM here
----------------------------
The IRM heterogeneity path (``_heterogeneity.py``) estimates each *specific*
treatment on the single-method subsample (boil-only vs none, dropping every
household using another method).  That subsample suffers positivity failure —
country FE predict adoption almost perfectly (cf_d ~ 1) — and the IRM
specific-treatment contrasts disagree wildly with the full-sample APOS (even
flipping sign for chlorine).  APOS is the correct estimand: one multi-class
propensity P(D=d | X) on the FULL sample, AIPW potential outcomes E[Y(d)], and
the contrast ATE(d vs 0) = E[Y(d)] - E[Y(0)].  This module projects that same
machinery onto moderators.

Two distinct objects (do NOT conflate)
--------------------------------------
* **GAPO / CAPO  (levels)** — ``DoubleMLAPO.gapo`` / ``.capo``, native.
  E[Y(d) | group] and E[Y(d) | x): the counterfactual outcome *level* the group
  would have if everyone were assigned treatment d.  For binary Y, a probability.
  NOT a treatment effect.
* **GATE / CATE  (effects)** — the contrast E[Y(d) - Y(0) | .].  Obtained by
  forming the per-observation AIPW *contrast signal* psi_b(d) - psi_b(0) and
  running ONE ``DoubleMLBLP`` on it (the same projection IRM.gate/.cate do
  internally).  This gives the correct point estimate AND valid pointwise / joint
  (uniform, Semenova-Chernozhukov) bands.  Differencing two separate gapo objects
  recovers the point estimate but NOT a valid CI (their covariance is lost), so
  effects always go through the contrast-signal path.

Inference notes
---------------
* ``DoubleMLAPOS`` is fit i.i.d. (no clustered SE in doubleml 0.11.x); the U5
  child data is mostly singletons so the clustered/i.i.d. gap is negligible
  (see ``_apos.py``).  This mirrors the APOS headline table.
* ``n_rep`` is forced to 1: ``DoubleMLBLP`` / gate / cate require a single
  cross-fitting repetition, and the per-level APO ``psi`` arrays must align row
  for row across levels (one shared sample split).
"""

import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import doubleml as dml
from patsy import dmatrix, build_design_matrices
from sklearn.base import clone

from _config import N_FOLDS, MIN_OBSERVATIONS, CHECKPOINT_DIR, logger
from _data import create_model_matrix
from _learners import create_learners_for_binary
from _apos import _build_treat_cat, APOS_LEVEL_LABELS, APOS_MIN_LEVEL_N


# =============================================================================
# Checkpointing — fitting the base APOS is the expensive step; cache the fitted
# object so re-runs skip it.  Delete the .pkl (or pass skip_checkpoint=True) to
# force a refit, e.g. after changing confounders / folds / learner definitions.
# =============================================================================

def load_or_fit(cp_file, build_fn: Callable[[], Any], skip_checkpoint: bool = False):
    """Return a cached pickle if present, else build, cache, and return it.

    ``build_fn`` returns the object to cache (or None to skip caching).
    """
    if not skip_checkpoint and cp_file.is_file():
        try:
            with open(cp_file, "rb") as f:
                obj = pickle.load(f)
            logger.info(f"    Loaded checkpoint: {cp_file.name}")
            return obj
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    Could not load checkpoint {cp_file.name}: {exc}")
    obj = build_fn()
    if not skip_checkpoint and obj is not None:
        try:
            with open(cp_file, "wb") as f:
                pickle.dump(obj, f)
            logger.info(f"    Saved checkpoint: {cp_file.name}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    Could not save checkpoint {cp_file.name}: {exc}")
    return obj


# =============================================================================
# Base APOS fit (shared by all heterogeneity projections)
# =============================================================================

def fit_base_apos(
    dt: pd.DataFrame,
    outcome_var: str,
    dataset_type: str = "HH",
    learner_name: str = "stacked",
    confounder_groups: Optional[Dict[str, List[str]]] = None,
    reference: int = 0,
    n_folds: int = N_FOLDS,
    skip_checkpoint: bool = False,
) -> Optional[Tuple[dml.DoubleMLAPOS, pd.DataFrame, List[int]]]:
    """Fit a single DoubleMLAPOS (n_rep=1) for heterogeneity projection.

    Mirrors ``_apos.estimate_apos`` data prep (complete-case mask + sparse-level
    drop) but keeps a row-aligned ``dt_clean`` so moderator columns can be pulled
    in the SAME order as the per-level ``psi`` arrays.  The fitted (apos,
    dt_clean, levels) tuple is checkpointed per (dataset, outcome, learner);
    delete the .pkl or pass ``skip_checkpoint=True`` to force a refit.

    Returns
    -------
    (apos, dt_clean, levels) : fitted APOS, the aligned complete-case DataFrame,
        and the kept treatment levels (sorted, including the reference).
        None on insufficient data.
    """
    cp_file = CHECKPOINT_DIR / f"apos_het_{dataset_type}_{outcome_var}_{learner_name}.pkl"

    def _build():
        return _fit_base_apos_impl(
            dt, outcome_var, dataset_type, learner_name,
            confounder_groups, reference, n_folds,
        )

    return load_or_fit(cp_file, _build, skip_checkpoint=skip_checkpoint)


def _fit_base_apos_impl(
    dt, outcome_var, dataset_type, learner_name,
    confounder_groups, reference, n_folds,
):
    """Actual fit (no checkpoint); see ``fit_base_apos``."""
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
        logger.error(f"    APOS(het) {outcome_var}: too few complete obs (N={n})")
        return None

    X = X[mask.values]
    dt_clean = dt.loc[mask].reset_index(drop=True)
    y = dt_clean[outcome_var].to_numpy(dtype=float)
    d = dt_clean["treat_cat"].astype(int).to_numpy()

    # Drop sparse treatment levels (keep the reference regardless), aligning dt_clean.
    counts = pd.Series(d).value_counts()
    levels = sorted(
        int(lv) for lv in counts.index
        if counts[lv] >= APOS_MIN_LEVEL_N or lv == reference
    )
    dropped = [int(lv) for lv in counts.index if int(lv) not in levels]
    if dropped:
        logger.warning(
            f"    APOS(het) {outcome_var}: dropping sparse levels "
            f"{[APOS_LEVEL_LABELS.get(lv, lv) for lv in dropped]} (< {APOS_MIN_LEVEL_N} obs)"
        )
        keep = np.isin(d, levels)
        X, y, d = X[keep], y[keep], d[keep]
        dt_clean = dt_clean.loc[keep].reset_index(drop=True)
        n = int(keep.sum())
    if reference not in levels or len(levels) < 2:
        logger.error(f"    APOS(het) {outcome_var}: insufficient treatment levels {levels}")
        return None

    lset = create_learners_for_binary()[learner_name]
    data = dml.DoubleMLData.from_arrays(x=X, y=y, d=d)
    apos = dml.DoubleMLAPOS(
        obj_dml_data=data,
        ml_g=clone(lset["g"]),
        ml_m=clone(lset["m"]),
        treatment_levels=levels,
        n_folds=n_folds,
        n_rep=1,  # forced: DoubleMLBLP / gate / cate require a single rep
        trimming_rule="truncate",
        trimming_threshold=0.01,
    )
    logger.info(
        f"    APOS(het) {outcome_var} [{dataset_type}] learner={learner_name} "
        f"levels={levels} N={n:,} (i.i.d.) ..."
    )
    apos.fit()
    logger.info("      base APOS E[Y(d)] = " + ", ".join(
        f"{APOS_LEVEL_LABELS.get(lv, lv)}={c:+.4f}"
        for lv, c in zip(levels, np.ravel(apos.coef))
    ))
    return apos, dt_clean, levels


# =============================================================================
# Contrast signal: AIPW pseudo-outcome for ATE(d vs reference)
# =============================================================================

def _level_signal(apo: Any) -> np.ndarray:
    """Per-observation AIPW signal whose mean is E[Y(d)] (the APO point estimate).

    From the APO orthogonal score psi = psi_a * theta + psi_b (with psi_a = -1),
    the efficient signal is -psi_b / psi_a; coded elementwise so it is correct
    regardless of any per-observation weighting.
    """
    pe = apo.psi_elements
    psi_a = np.asarray(pe["psi_a"]).reshape(-1)
    psi_b = np.asarray(pe["psi_b"]).reshape(-1)
    return psi_b / (-psi_a)


def _contrast_signal(
    apos: dml.DoubleMLAPOS, levels: List[int], level: int, reference: int = 0
) -> np.ndarray:
    """AIPW contrast signal for ATE(level vs reference); mean = the APOS contrast."""
    apo_d = apos.modellist[levels.index(level)]
    apo_0 = apos.modellist[levels.index(reference)]
    return _level_signal(apo_d) - _level_signal(apo_0)


# =============================================================================
# GATE — group average treatment effect (contrast, with valid bands)
# =============================================================================

def estimate_gate_apos(
    apos: dml.DoubleMLAPOS,
    levels: List[int],
    level: int,
    group_series: pd.Series,
    group_labels: Optional[Dict[Any, str]] = None,
    reference: int = 0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Project the ATE(level vs reference) contrast signal onto group indicators.

    Returns one row per group: coef, pointwise CI, and JOINT (uniform) CI — same
    schema as ``_heterogeneity.estimate_gate``.
    """
    sig = _contrast_signal(apos, levels, level, reference)
    g = group_series.reset_index(drop=True)
    groups = pd.get_dummies(g, prefix="grp")
    groups = groups.loc[:, groups.sum(axis=0) > 0]  # mutually exclusive, drop empty

    blp = dml.DoubleMLBLP(sig, basis=groups, is_gate=True).fit()
    ci_point = blp.confint(joint=False, level=ci_level)
    ci_joint = blp.confint(joint=True, level=ci_level)

    rows = []
    for i, col in enumerate(groups.columns):
        raw = col.replace("grp_", "")
        label = None
        if group_labels is not None:
            for k, v in group_labels.items():
                if str(k) == raw:
                    label = v
        rows.append({
            "group_value": raw,
            "group_label": label if label is not None else raw,
            "n": int(groups[col].sum()),
            "coef": float(ci_point.iloc[i, 1]),
            "ci_lower": float(ci_point.iloc[i, 0]),
            "ci_upper": float(ci_point.iloc[i, -1]),
            "ci_lower_joint": float(ci_joint.iloc[i, 0]),
            "ci_upper_joint": float(ci_joint.iloc[i, -1]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# CATE — continuous conditional treatment effect (contrast, spline basis)
# =============================================================================

def estimate_cate_apos_spline(
    apos: dml.DoubleMLAPOS,
    levels: List[int],
    level: int,
    x_series: pd.Series,
    reference: int = 0,
    df_spline: int = 5,
    degree: int = 3,
    n_grid: int = 100,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Project the ATE(level vs reference) contrast signal onto a B-spline basis.

    Returns a grid DataFrame: x, cate, pointwise CI, JOINT CI — same schema as
    ``_heterogeneity.estimate_cate_spline`` (so ``figures.plot_cate_curves``
    consumes it unchanged).
    """
    sig = _contrast_signal(apos, levels, level, reference)
    x = x_series.reset_index(drop=True).astype(float)
    formula = f"bs(x, df={df_spline}, degree={degree}, include_intercept=True)"
    design = dmatrix(formula, {"x": x.values}, return_type="dataframe")

    blp = dml.DoubleMLBLP(sig, basis=design, is_gate=False).fit()

    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    grid_design = build_design_matrices([design.design_info], {"x": x_grid})[0]
    grid_df = pd.DataFrame(np.asarray(grid_design), columns=design.columns)

    ci_point = blp.confint(grid_df, joint=False, level=ci_level)
    ci_joint = blp.confint(grid_df, joint=True, level=ci_level)

    return pd.DataFrame({
        "x": x_grid,
        "cate": ci_point.iloc[:, 1].to_numpy(),
        "ci_lower": ci_point.iloc[:, 0].to_numpy(),
        "ci_upper": ci_point.iloc[:, 2].to_numpy(),
        "ci_lower_joint": ci_joint.iloc[:, 0].to_numpy(),
        "ci_upper_joint": ci_joint.iloc[:, 2].to_numpy(),
    })


# =============================================================================
# GAPO / CAPO — native potential-outcome LEVELS (E[Y(d) | .], not effects)
# =============================================================================

def estimate_gapo_levels(
    apos: dml.DoubleMLAPOS,
    levels: List[int],
    group_series: pd.Series,
    group_labels: Optional[Dict[Any, str]] = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Native GAPO: E[Y(d) | group] for every treatment level d.

    Returns one row per (level, group) — counterfactual outcome LEVELS, not
    effects.  Use these for the "predicted contamination under each treatment"
    story; use ``estimate_gate_apos`` for the effect with valid CIs.
    """
    g = group_series.reset_index(drop=True)
    groups = pd.get_dummies(g, prefix="grp")
    groups = groups.loc[:, groups.sum(axis=0) > 0]

    rows = []
    for lv in levels:
        apo = apos.modellist[levels.index(lv)]
        blp = apo.gapo(groups)
        ci_point = blp.confint(joint=False, level=ci_level)
        ci_joint = blp.confint(joint=True, level=ci_level)
        for i, col in enumerate(groups.columns):
            raw = col.replace("grp_", "")
            label = None
            if group_labels is not None:
                for k, v in group_labels.items():
                    if str(k) == raw:
                        label = v
            rows.append({
                "treatment_level": lv,
                "treatment_label": APOS_LEVEL_LABELS.get(lv, str(lv)),
                "group_value": raw,
                "group_label": label if label is not None else raw,
                "n": int(groups[col].sum()),
                "gapo": float(ci_point.iloc[i, 1]),
                "ci_lower": float(ci_point.iloc[i, 0]),
                "ci_upper": float(ci_point.iloc[i, -1]),
                "ci_lower_joint": float(ci_joint.iloc[i, 0]),
                "ci_upper_joint": float(ci_joint.iloc[i, -1]),
            })
    return pd.DataFrame(rows)


def estimate_capo_spline(
    apos: dml.DoubleMLAPOS,
    levels: List[int],
    x_series: pd.Series,
    df_spline: int = 5,
    degree: int = 3,
    n_grid: int = 100,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Native CAPO: smooth E[Y(d) | x] curve for every treatment level d.

    Returns a long grid DataFrame (one block per level) of counterfactual
    outcome LEVELS over x.
    """
    x = x_series.reset_index(drop=True).astype(float)
    formula = f"bs(x, df={df_spline}, degree={degree}, include_intercept=True)"
    design = dmatrix(formula, {"x": x.values}, return_type="dataframe")
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    grid_design = build_design_matrices([design.design_info], {"x": x_grid})[0]
    grid_df = pd.DataFrame(np.asarray(grid_design), columns=design.columns)

    blocks = []
    for lv in levels:
        apo = apos.modellist[levels.index(lv)]
        cap = apo.capo(design)
        ci_point = cap.confint(grid_df, joint=False, level=ci_level)
        ci_joint = cap.confint(grid_df, joint=True, level=ci_level)
        blocks.append(pd.DataFrame({
            "treatment_level": lv,
            "treatment_label": APOS_LEVEL_LABELS.get(lv, str(lv)),
            "x": x_grid,
            "capo": ci_point.iloc[:, 1].to_numpy(),
            "ci_lower": ci_point.iloc[:, 0].to_numpy(),
            "ci_upper": ci_point.iloc[:, 2].to_numpy(),
            "ci_lower_joint": ci_joint.iloc[:, 0].to_numpy(),
            "ci_upper_joint": ci_joint.iloc[:, 2].to_numpy(),
        }))
    return pd.concat(blocks, ignore_index=True)
