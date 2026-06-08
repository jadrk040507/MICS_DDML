"""
Engine for DML_Final_Analysis.ipynb.

This module is the single source of methodological truth for the pedagogical
notebook.  It reuses the production package (``final/``) for the parts that must
be correct and identical across the two code paths:

  * the 7 ML learners (OLS, Lasso, Ridge, ENet, RF, XGB, Stacked), all built
    with ``create_learners_for_binary`` so that the outcome nuisance ``ml_g``
    returns probabilities bounded in [0, 1] (via ``ProbaRegressor``);
  * PSU-clustered standard errors with the same 3-attempt fallback used by
    ``models.estimate_effect`` (default folds + clusters -> 2 folds + clusters
    -> 2 folds, no clusters);
  * structured sensitivity extraction (``sensitivity_params``), not regex.

It adds notebook-only extensions (APOS multi-treatment with the stacked
ensemble and proper ``causal_contrast`` standard errors, GATE) and writers that
emit publication LaTeX tables to ``Output/tables/``, reusing
``final/tables.create_main_table`` for the headline IRM tables so the layout is
identical to the production pipeline.

All estimation is clustered, bounded, and uses the same n_folds / n_rep as the
package.  Import this from the notebook with::

    import sys; sys.path.insert(0, '../final'); sys.path.insert(0, '.')
    import nb_dml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence

import numpy as np
import pandas as pd
import doubleml as dml
from scipy.stats import norm as _norm
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV

# --- wire in the production package -------------------------------------------
_FINAL = Path(__file__).resolve().parents[1] / "final"
if str(_FINAL) not in sys.path:
    sys.path.insert(0, str(_FINAL))

from config import (  # noqa: E402  (path set above)
    N_FOLDS, N_REP, RANDOM_STATE, N_JOBS, CLUSTER_VAR,
    OUTPUT_DIR, LEARNER_LABELS,
)
from learners import (  # noqa: E402
    create_learners_for_binary,
    _lasso_classifier, _ridge_classifier, _rf_classifier,
)
import tables as pkg_tables  # noqa: E402

TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Sensitivity defaults (match models.estimate_effect)
SENS_CF = 0.03
SENS_RHO = 1.0
SENS_LEVEL = 0.95

LEARNER_ORDER = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]


# =============================================================================
# LEARNERS
# =============================================================================

def get_learners() -> Dict[str, Dict[str, Any]]:
    """7 learners with bounded (ProbaRegressor) ml_g, classifier ml_m.

    Identical to the production package; see ``learners.create_learners_for_binary``.
    """
    return create_learners_for_binary()


def _stacked_multiclass_m() -> StackingClassifier:
    """Multi-class propensity learner for APOS (P(D=d|X), d in 0..4).

    Base learners are the same family used elsewhere (penalised logit + RF),
    all natively multi-class; meta-learner is a multinomial logit.  This keeps
    the APOS treatment model on the *stacked ensemble*, consistent with the IRM
    headline learner.
    """
    final_m = LogisticRegressionCV(
        cv=3, solver="lbfgs", max_iter=1000, n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    return StackingClassifier(
        estimators=[
            ("lasso", _lasso_classifier(n_jobs=1)),
            ("ridge", _ridge_classifier(n_jobs=1)),
            ("rf", _rf_classifier(n_jobs=1)),
        ],
        final_estimator=final_m,
        cv=3, n_jobs=N_JOBS,
    )


# =============================================================================
# CLUSTERED IRM  (mirrors models.estimate_effect fit strategy)
# =============================================================================

def _is_cluster_split_error(exc: Exception) -> bool:
    """True for errors that the 'drop clustering' fallback should swallow.

    Covers degenerate cluster-fold splits (IRM) and the fact that
    ``DoubleMLAPOS`` does not support clustered data in doubleml 0.11.x
    (raises AttributeError on ``_n_folds_per_cluster``).
    """
    msg = str(exc).lower()
    return any(s in msg for s in (
        "zero-dimensional", "all_smpls_cluster", "empty",
        "_n_folds_per_cluster", "cluster",
    ))


def analysis_frame(
    data: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: Sequence[str],
) -> pd.DataFrame:
    """Complete-case analysis frame carrying the PSU cluster id when present.

    Selects [y, d, *x, Cluster_var], drops missing rows, returns float frame
    (Cluster_var kept as-is for grouping).
    """
    cols = [y_col, d_col] + list(x_cols)
    if CLUSTER_VAR in data.columns and CLUSTER_VAR not in cols:
        cols = cols + [CLUSTER_VAR]
    out = data[cols].dropna().reset_index(drop=True)
    keep_float = [c for c in out.columns if c != CLUSTER_VAR]
    out[keep_float] = out[keep_float].astype(float)
    return out


def fit_irm(
    df: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: Sequence[str],
    learner: Dict[str, Any],
    use_cluster: bool = True,
    n_folds: int = N_FOLDS,
    n_rep: int = N_REP,
    sensitivity: bool = True,
) -> Optional[Dict[str, Any]]:
    """Fit one DoubleMLIRM with PSU clustering + 3-attempt fallback.

    Builds ``DoubleMLData`` from a *named* DataFrame (with ``cluster_cols``) so
    confounder names survive into ``sensitivity_benchmark``.  Clustering is taken
    from the ``Cluster_var`` column when present and ``use_cluster`` is True;
    the fallback drops folds then clustering if cluster-fold splits degenerate.

    Returns a result dict with the same keys as ``models.estimate_effect`` (so
    ``final/tables.create_main_table`` can consume it directly), or None.
    """
    x_cols = list(x_cols)
    has_cluster = use_cluster and (CLUSTER_VAR in df.columns)
    keep = [y_col, d_col] + x_cols + ([CLUSTER_VAR] if has_cluster else [])
    work = df[keep].dropna().reset_index(drop=True)

    def _fit(nf, with_cluster):
        data = dml.DoubleMLData(
            work, y_col=y_col, d_cols=d_col, x_cols=x_cols,
            cluster_cols=(CLUSTER_VAR if with_cluster else None),
        )
        model = dml.DoubleMLIRM(
            obj_dml_data=data,
            ml_g=clone(learner["g"]),
            ml_m=clone(learner["m"]),
            n_folds=nf, n_rep=n_rep, score="ATE",
            trimming_rule="truncate", trimming_threshold=0.01,
            draw_sample_splitting=True,
        )
        model.fit()
        return model

    model, clustered = None, False
    for nf, wc in [(n_folds, has_cluster), (2, has_cluster), (2, False)]:
        try:
            model = _fit(nf, wc)
            clustered = wc
            break
        except Exception as exc:  # noqa: BLE001
            if _is_cluster_split_error(exc):
                continue
            raise
    if model is None:
        return None

    ci = model.confint()
    ci_lower = float(ci.iloc[0, 0]) if hasattr(ci, "iloc") else float(ci[0, 0])
    ci_upper = float(ci.iloc[0, 1]) if hasattr(ci, "iloc") else float(ci[0, 1])

    rv = rva = None
    if sensitivity:
        try:
            model.sensitivity_analysis(cf_y=SENS_CF, cf_d=SENS_CF, rho=SENS_RHO, level=SENS_LEVEL)
            sp = model.sensitivity_params
            if sp is not None:
                rv = float(np.ravel(sp["rv"])[0])
                rva = float(np.ravel(sp["rva"])[0])
        except Exception:  # noqa: BLE001
            pass

    return {
        "outcome": y_col,
        "treatment": d_col,
        "learner": None,            # filled by caller
        "subgroup_var": None,
        "subgroup_val": None,
        "dataset_type": None,       # filled by caller
        "coef": float(model.coef[0]),
        "se": float(model.se[0]),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": int(len(work)),
        "n_treated": int(work[d_col].sum()),
        "n_clusters": int(work[CLUSTER_VAR].nunique()) if clustered else None,
        "clustered": clustered,
        "rv_q": rv,
        "rv_qa": rva,
        "model": model,
    }


def fit_irm_all_learners(
    df: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: Sequence[str],
    dataset_type: str,
    learners: Optional[Dict[str, Dict[str, Any]]] = None,
    use_cluster: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Fit all 7 learners for one (outcome, treatment); return result dicts.

    ``df`` should already be the complete-case frame and (if clustering) carry
    the ``Cluster_var`` column.
    """
    if learners is None:
        learners = get_learners()
    out = []
    for name in LEARNER_ORDER:
        if name not in learners:
            continue
        if verbose:
            print(f"  {LEARNER_LABELS.get(name, name):<14}", end=" ", flush=True)
        res = fit_irm(df, y_col, d_col, x_cols, learners[name], use_cluster=use_cluster)
        if res is None:
            if verbose:
                print("FAILED")
            continue
        res["learner"] = name
        res["dataset_type"] = dataset_type
        out.append(res)
        if verbose:
            star = "*" if (res["ci_lower"] > 0 or res["ci_upper"] < 0) else " "
            print(f"ATE={res['coef']:+.4f} ({res['se']:.4f}) {star}")
    return out


def results_to_frame(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tidy DataFrame from IRM result dicts (for display / coefplots)."""
    rows = []
    for r in results:
        se = r["se"]
        t = r["coef"] / se if se else np.nan
        rows.append({
            "Learner": LEARNER_LABELS.get(r["learner"], r["learner"]),
            "learner": r["learner"],
            "ATE": r["coef"], "Std Error": se, "t-stat": t,
            "p-value": 2 * _norm.sf(abs(t)) if se else np.nan,
            "CI Lower": r["ci_lower"], "CI Upper": r["ci_upper"],
            "RV": r.get("rv_q"), "RVa": r.get("rv_qa"), "N": r["n"],
        })
    return pd.DataFrame(rows)


def get_model(results: List[Dict[str, Any]], learner: str = "stacked"):
    """Return the fitted DoubleMLIRM object for a given learner from results."""
    for r in results:
        if r["learner"] == learner:
            return r.get("model")
    return None


# =============================================================================
# APOS  (multi-treatment, stacked ensemble, proper contrast SEs)
# =============================================================================

def fit_apos(
    df: pd.DataFrame,
    y_col: str,
    d_cat_col: str,
    x_cols: Sequence[str],
    dataset_type: str,
    level_labels: Dict[int, str],
    reference: int = 0,
    use_cluster: bool = True,
    n_folds: int = N_FOLDS,
    n_rep: int = N_REP,
) -> Dict[str, Any]:
    """Joint APOS on the full sample with the stacked ensemble.

    ml_g = stacked ProbaRegressor ensemble (bounded E[Y|D=d,X]);
    ml_m = stacked multi-class propensity P(D=d|X).
    ATE(d vs ref) and SEs come from ``causal_contrast`` (NOT the conservative
    sqrt(V_d+V_0) approximation), plus contrast-level RV/RVa.

    Note: ``DoubleMLAPOS`` does not support clustering in doubleml 0.11.x, so the
    fallback drops clustering automatically; ``clustered`` is returned as False.
    """
    x_cols = list(x_cols)
    has_cluster = use_cluster and (CLUSTER_VAR in df.columns)
    keep = [y_col, d_cat_col] + x_cols + ([CLUSTER_VAR] if has_cluster else [])
    work = df[keep].dropna().reset_index(drop=True)
    work[d_cat_col] = work[d_cat_col].astype(int)
    levels = sorted(int(v) for v in work[d_cat_col].unique())

    ml_g = get_learners()["stacked"]["g"]   # StackingRegressor of ProbaRegressor
    ml_m = _stacked_multiclass_m()

    def _fit(nf, with_cluster):
        data = dml.DoubleMLData(
            work, y_col=y_col, d_cols=d_cat_col, x_cols=x_cols,
            cluster_cols=(CLUSTER_VAR if with_cluster else None),
        )
        model = dml.DoubleMLAPOS(
            obj_dml_data=data,
            ml_g=clone(ml_g), ml_m=clone(ml_m),
            treatment_levels=levels,
            n_folds=nf, n_rep=n_rep,
            trimming_rule="truncate", trimming_threshold=0.01,
        )
        model.fit()
        return model

    apos = None
    clustered = False
    for nf, wc in [(n_folds, has_cluster), (2, has_cluster), (2, False)]:
        try:
            apos = _fit(nf, wc)
            clustered = wc
            break
        except Exception as exc:  # noqa: BLE001
            if _is_cluster_split_error(exc):
                continue
            raise
    if apos is None:
        raise RuntimeError(f"APOS failed for {y_col}")

    contrast = apos.causal_contrast(reference_levels=reference)
    csum = contrast.summary
    coefs = csum["coef"].to_numpy()
    ses = csum["std err"].to_numpy()
    ci_lo = csum.iloc[:, csum.columns.get_loc("2.5 %")].to_numpy()
    ci_hi = csum.iloc[:, csum.columns.get_loc("97.5 %")].to_numpy()

    # contrast-level robustness values
    rv = rva = None
    try:
        contrast.sensitivity_analysis(cf_y=SENS_CF, cf_d=SENS_CF, rho=SENS_RHO)
        sp = contrast.sensitivity_params
        if sp is not None:
            rv = np.asarray(sp["rv"], dtype=float).ravel()
            rva = np.asarray(sp["rva"], dtype=float).ravel()
    except Exception:  # noqa: BLE001
        pass

    rows = []
    non_ref = [lv for lv in levels if lv != reference]
    for i, lv in enumerate(non_ref):
        rows.append({
            "outcome": y_col,
            "dataset_type": dataset_type,
            "treatment_level": lv,
            "treatment": level_labels.get(lv, str(lv)),
            "coef": float(coefs[i]),
            "se": float(ses[i]),
            "ci_lower": float(ci_lo[i]),
            "ci_upper": float(ci_hi[i]),
            "rv_q": float(rv[i]) if rv is not None and i < len(rv) else None,
            "rv_qa": float(rva[i]) if rva is not None and i < len(rva) else None,
            "n": int(len(work)),
            "n_treated": int((work[d_cat_col] == lv).sum()),
        })
    return {"apos": apos, "contrast": contrast, "rows": rows,
            "levels": levels, "clustered": clustered}


# =============================================================================
# GATE  (group ATE from a fitted stacked IRM)
# =============================================================================

def make_groups(n: int, specs: List[tuple]) -> pd.DataFrame:
    """Mutually-exclusive, exhaustive bool group frame from (label, mask).

    Columns are bool dtype (required by DoubleMLIRM.gate); the last group
    absorbs any stragglers so the partition is exhaustive.
    """
    g = pd.DataFrame(index=range(n))
    assigned = np.zeros(n, dtype=bool)
    cols = []
    for label, mask in specs:
        m = np.asarray(mask, dtype=bool) & ~assigned
        g[label] = m
        assigned |= m
        cols.append(label)
    g[cols[-1]] = g[cols[-1]].to_numpy() | (~assigned)
    return g.astype(bool)


def gate_summary(irm_model, groups: pd.DataFrame) -> pd.DataFrame:
    """GATE summary with CI columns normalised to '2.5 %'/'97.5 %'.

    DoubleMLIRM.gate() returns statsmodels-style '[0.025'/'0.975]' CI columns;
    we rename them to match DoubleML's own .summary so downstream table/plot
    code can use a single convention.
    """
    gate = irm_model.gate(groups=groups)
    gs = gate.summary() if callable(gate.summary) else gate.summary
    return gs.rename(columns={"[0.025": "2.5 %", "0.975]": "97.5 %"})


# =============================================================================
# LaTeX WRITERS
# =============================================================================

def _stars(coef: float, se: float) -> str:
    if se == 0 or se != se:
        return ""
    t = abs(coef / se)
    return "***" if t > 2.576 else "**" if t > 1.960 else "*" if t > 1.645 else ""


def write_main_irm_tables(results: List[Dict[str, Any]]) -> List[Path]:
    """Headline IRM tables (outcome x treatment rows, learner columns).

    Reuses ``final/tables.create_main_table`` so layout matches the production
    pipeline exactly.  Emits one file per dataset (HH / U5).
    """
    paths = []
    for dtype, fname in [("HH", "tables/nb_main_irm_hh.tex"),
                         ("U5", "tables/nb_main_irm_u5.tex")]:
        if any(r.get("dataset_type") == dtype for r in results):
            paths.append(pkg_tables.create_main_table(results, filename=fname, dataset_type=dtype))
    return paths


def write_robustness_table(
    base_results: List[Dict[str, Any]],
    robust_results: List[Dict[str, Any]],
    filename: str = "tables/nb_robustness_hygiene.tex",
) -> Path:
    """Base vs. +hygiene specification (stacked learner) per outcome."""
    def _pick(res):
        return {(r["outcome"]): r for r in res if r["learner"] == "stacked"}
    b, rob = _pick(base_results), _pick(robust_results)
    outcomes = [o for o in b if o in rob]

    L = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Robustness: Adding Hygiene Controls (Stacked Ensemble)}",
         "\\label{tab:nb_robustness}",
         "\\begin{tabular}{lcccc}", "\\hline\\hline",
         "Outcome & ATE (Base) & ATE ($+$ Hygiene) & $\\Delta$ & N \\\\", "\\hline"]
    for o in outcomes:
        rb, rr = b[o], rob[o]
        lab = pkg_tables._get_outcome_label(o)
        delta = rr["coef"] - rb["coef"]
        L.append(
            f"{lab} & {rb['coef']:.3f}{_stars(rb['coef'], rb['se'])} "
            f"& {rr['coef']:.3f}{_stars(rr['coef'], rr['se'])} "
            f"& {delta:+.3f} & {rr['n']:,} \\\\"
        )
        L.append(f" & ({rb['se']:.3f}) & ({rr['se']:.3f}) & & \\\\")
    L += ["\\hline\\hline", "\\end{tabular}",
          "\\begin{minipage}{\\textwidth}\\footnotesize",
          "\\textit{Notes:} Stacked-ensemble DDML IRM, ATE score, PSU-clustered SEs. ",
          "Hygiene controls: soap-and-water availability, covered water storage. ",
          "$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.",
          "\\end{minipage}", "\\end{table}"]
    return _write(L, filename)


def write_specific_table(
    apos_rows: List[Dict[str, Any]],
    filename: str = "tables/nb_specific_treatments.tex",
    clustered: bool = False,
) -> Path:
    """APOS contrasts: each specific treatment vs. no treatment, all outcomes."""
    outcomes = []
    for r in apos_rows:
        if r["outcome"] not in outcomes:
            outcomes.append(r["outcome"])

    L = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Specific Treatment Effects vs. No Treatment (APOS, Stacked Ensemble)}",
         "\\label{tab:nb_specific}",
         "\\begin{tabular}{llccc}", "\\hline\\hline",
         "Outcome & Treatment & ATE & 95\\% CI & N (treated) \\\\", "\\hline"]
    for o in outcomes:
        rows = [r for r in apos_rows if r["outcome"] == o]
        for i, r in enumerate(rows):
            olab = pkg_tables._get_outcome_label(o) if i == 0 else ""
            L.append(
                f"{olab} & {r['treatment']} "
                f"& {r['coef']:.3f}{_stars(r['coef'], r['se'])} "
                f"& [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] "
                f"& {r['n_treated']:,} \\\\"
            )
            L.append(f" & & ({r['se']:.3f}) & & \\\\")
        if o != outcomes[-1]:
            L.append("\\addlinespace")
    L += ["\\hline\\hline", "\\end{tabular}",
          "\\begin{minipage}{\\textwidth}\\footnotesize",
          "\\textit{Notes:} Average Potential Outcome (AIPW) contrasts, ATE(d) $=$ E[Y(d)]$-$E[Y(0)], ",
          "estimated jointly on the full sample. Stacked ensemble for both nuisances; ",
          "multi-class propensity P(D$=$d$\\mid$X). SEs from \\texttt{causal\\_contrast}"
          + (", PSU-clustered. " if clustered
             else " (i.i.d.; DoubleMLAPOS does not support clustering). "),
          "$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.",
          "\\end{minipage}", "\\end{table}"]
    return _write(L, filename)


def write_sensitivity_table(
    results: List[Dict[str, Any]],
    filename: str = "tables/nb_sensitivity.tex",
) -> Path:
    """RV / RVa for the stacked IRM (any treatment), all outcomes."""
    rows = [r for r in results if r["learner"] == "stacked" and r.get("rv_q") is not None]
    L = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Sensitivity to Unobserved Confounding (Stacked IRM)}",
         "\\label{tab:nb_sensitivity}",
         "\\begin{tabular}{lcccc}", "\\hline\\hline",
         "Outcome & Treatment & ATE & RV (\\%) & RV$_\\alpha$ (\\%) \\\\", "\\hline"]
    for r in rows:
        L.append(
            f"{pkg_tables._get_outcome_label(r['outcome'])} "
            f"& {pkg_tables._get_treatment_label(r['treatment'])} "
            f"& {r['coef']:.3f}{_stars(r['coef'], r['se'])} "
            f"& {r['rv_q'] * 100:.1f} & {r['rv_qa'] * 100:.1f} \\\\"
        )
    L += ["\\hline\\hline", "\\end{tabular}",
          "\\begin{minipage}{\\textwidth}\\footnotesize",
          "\\textit{Notes:} Chernozhukov et al. (2022) sensitivity. RV is the strength "
          "(share of residual variance in both treatment and outcome) an unobserved "
          "confounder would need to drive the point estimate to zero; RV$_\\alpha$ to bring "
          "the 95\\% CI to zero. Worst case $\\rho=1$.",
          "\\end{minipage}", "\\end{table}"]
    return _write(L, filename)


def write_gate_table(
    gate_data: Dict[str, Dict[str, Any]],
    filename: str = "tables/nb_gate.tex",
) -> Path:
    """GATE forest data -> table. ``gate_data`` = {outcome: {group_name: {summary, labels}}}."""
    L = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Group Average Treatment Effects (Stacked IRM, Any Treatment)}",
         "\\label{tab:nb_gate}",
         "\\begin{tabular}{lllcc}", "\\hline\\hline",
         "Outcome & Grouping & Group & GATE & 95\\% CI \\\\", "\\hline"]
    outcomes = list(gate_data.keys())
    for o in outcomes:
        first_o = True
        for gname, gd in gate_data[o].items():
            gs, labels = gd["summary"], gd["labels"]
            coefs = gs["coef"].to_numpy()
            lo = gs.iloc[:, gs.columns.get_loc("2.5 %")].to_numpy()
            hi = gs.iloc[:, gs.columns.get_loc("97.5 %")].to_numpy()
            ses = gs["std err"].to_numpy() if "std err" in gs.columns else np.full(len(coefs), np.nan)
            for i, lab in enumerate(labels):
                ocell = pkg_tables._get_outcome_label(o) if first_o else ""
                gcell = gname if i == 0 else ""
                first_o = False
                L.append(
                    f"{ocell} & {gcell} & {lab} "
                    f"& {coefs[i]:.3f}{_stars(coefs[i], ses[i])} "
                    f"& [{lo[i]:.3f}, {hi[i]:.3f}] \\\\"
                )
            L.append("\\addlinespace")
        if o != outcomes[-1]:
            L.append("\\midrule")
    L += ["\\hline\\hline", "\\end{tabular}",
          "\\begin{minipage}{\\textwidth}\\footnotesize",
          "\\textit{Notes:} GATE $=$ E[$\\tau(X)\\mid$ group]. Stacked-ensemble IRM, PSU-clustered. ",
          "$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.",
          "\\end{minipage}", "\\end{table}"]
    return _write(L, filename)


def _write(lines: List[str], filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Table saved to: {path}")
    return path
