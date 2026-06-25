"""
Publication-quality LaTeX tables for the MICS DDML Analysis.

Generates:
- create_results_table : combined IRM-on-top-of-APOS table (outcomes x learners),
  coef over (se), plus stacked base-learner meta-weights (g | m) in the stacked
  columns and RV / RVa sensitivity rows.

The grouped tables need a landscape/booktabs preamble; see the comment above
``_render_grouped_table``.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from _config import (
    OUTPUT_DIR, LEARNER_LABELS, LEARNER_NAMES,
    N_FOLDS, N_REP,
)


def _format_coef(coef: float, se: float, ci_lower: float = None, ci_upper: float = None) -> str:
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


def _format_rv(rv: float) -> str:
    """Format robustness value as percentage."""
    if rv is None or rv != rv:  # NaN check
        return "---"
    return f"{rv * 100:.1f}\\%"


def _get_outcome_label(outcome_var: str) -> str:
    """Get human-readable outcome label."""
    labels = {
        "SomeRiskHome": "Some Risk Home (E.coli 1--100 CFU)",
        "VeryHighRiskHome": "Very High Risk Home (E.coli $\\geq$ 101 CFU)",
        "diarrhea": "Diarrhea (under-5)",
    }
    return labels.get(outcome_var, outcome_var)


def _get_treatment_label(treatment_var: str) -> str:
    """Get human-readable treatment label."""
    labels = {
        "water_treatment": "Any Treatment",
        "treat_boil": "Boil",
        "treat_chlorine": "Chlorine",
        "treat_filter": "Filter",
        "treat_other": "Other",
    }
    return labels.get(treatment_var, treatment_var)





def create_sensitivity_table(
    results: List[Dict[str, Any]],
    filename: str = "table_sensitivity.tex",
    learner: str = "stacked",
) -> Path:
    """Sensitivity to unobserved confounding (RV / RVa) for the IRM headline learner.

    Consumes the same result dicts as ``create_main_table`` (rv_q / rv_qa are
    populated by ``models.estimate_effect``).
    """
    rows = [
        r for r in results
        if r.get("learner") == learner
        and r.get("subgroup_var") is None
        and r.get("rv_q") is not None
    ]

    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Sensitivity to Unobserved Confounding (Stacked IRM)}",
             "\\label{tab:sensitivity}",
             "\\begin{tabular}{llccc}", "\\hline\\hline",
             "Outcome & Treatment & ATE & RV & RV$_\\alpha$ \\\\", "\\hline"]
    for r in rows:
        lines.append(
            f"{_get_outcome_label(r['outcome'])} & {_get_treatment_label(r['treatment'])} "
            f"& {_format_coef(r['coef'], r['se'])} "
            f"& {_format_rv(r['rv_q'])} & {_format_rv(r['rv_qa'])} \\\\"
        )
    lines += ["\\hline\\hline", "\\end{tabular}",
              "\\begin{minipage}{\\textwidth}\\footnotesize",
              "\\textit{Notes:} Chernozhukov et al. (2022) sensitivity, worst case $\\rho=1$. "
              "RV is the share of residual variance in both treatment and outcome an unobserved "
              "confounder would need to drive the point estimate to zero; RV$_\\alpha$ to bring the "
              "95\\% CI to zero.",
              "\\end{minipage}", "\\end{table}"]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to: {filepath}")
    return filepath


# Confounder-group display label + glossary description, keyed by group name.
# The DISPLAY ORDER is NOT defined here — it is derived from BASE_CONFOUNDERS
# (+ U5_ADDITIONAL_CONFOUNDERS) at build time so the sensitivity table, the
# model spec, and the IRM/APOS confounder note can never drift apart.
_SENS_GROUP_META = {
    "wealth":       ("Wealth quintile",
                     r"household wealth-index quintiles (Q2--Q5 vs.\ Q1)"),
    "education":    ("Education",
                     r"education of the household head (primary, secondary, missing vs.\ none)"),
    "country":      ("Country FE",
                     r"country fixed effects"),
    "urban":        ("Urban",
                     r"urban vs.\ rural residence"),
    "water_source": ("Water source",
                     r"main drinking-water source (piped, borehole, protected/unprotected "
                     r"well or spring, surface/rain, packaged, other)"),
    "hh_demog":     ("HH demographics",
                     r"indicators for any under-5 child, a girl under~15, and a boy under~15"),
    "sanitation":   ("Sanitation",
                     r"toilet-facility type (pit latrine, open defecation, missing vs.\ flush)"),
    "wq27":         ("Source E.coli (decile)",
                     r"deciles of the source-water E.coli test (WQ27, CFU/100\,ml)"),
    "child":        ("Child age/sex",
                     r"child's age in years and sex (U5/diarrhea models only)"),
}


def _sens_groups():
    """Ordered [(key, label, desc), ...] following BASE_CONFOUNDERS (+ U5 child).

    Order is taken from the model spec so it stays in lock-step with the IRM /
    APOS confounder vector; any group lacking metadata falls back to its key.
    """
    from _config import BASE_CONFOUNDERS, U5_ADDITIONAL_CONFOUNDERS
    keys = list(BASE_CONFOUNDERS.keys()) + list(U5_ADDITIONAL_CONFOUNDERS.keys())
    out = []
    for k in keys:
        label, desc = _SENS_GROUP_META.get(k, (k, k))
        out.append((k, label, desc))
    return out


def create_sensitivity_benchmark_table(
    df,
    filename: str = "table_sensitivity_benchmark.tex",
    learner: str = "lasso",
    caption: str = None,
    label: str = None,
    landscape: bool = True,
) -> Path:
    """Confounder benchmarking, grouped landscape format (IRM/APOS style).

    Columns group the three outcomes; within each, three sub-columns report the
    benchmark for every observed confounder GROUP: cf_y (partial R^2 in the
    outcome), cf_d (partial R^2 in adoption), and d-theta (the bias an
    unobserved confounder *as strong as this group* would induce).  Rows are the
    treatments — Panel A any-treatment, Panel B each specific method vs none —
    with an ATE and an RV / RV_alpha header line per treatment.

    Consumes the long DataFrame from ``60_run_sensitivity.py`` (one row per
    spec x confounder group).  Requires a landscape/booktabs preamble.
    """
    sub = df[df["learner"] == learner].copy() if "learner" in df else df.copy()

    present = {r for r in sub["outcome"].unique()}
    outcomes = [o for o in _OUTCOME_ORDER if o in present] or sorted(present)
    O = len(outcomes)
    SUB = 3  # cf_y, cf_d, dtheta per outcome
    ncol = 1 + O * SUB

    # lookups: spec-level + per-group
    spec = {}   # (outcome, treatment) -> {ate, se, rv_q, rv_qa, n}
    cell = {}   # (outcome, treatment, group) -> {cf_y, cf_d, delta_theta}
    treats = set()
    for _, r in sub.iterrows():
        o = r["outcome"]
        if _DATASET_OF_OUTCOME.get(o) and r.get("dataset_type") != _DATASET_OF_OUTCOME[o]:
            continue
        t = r["treatment"]
        treats.add(t)
        spec.setdefault((o, t), {
            "ate": float(r["ate"]), "se": float(r["se"]),
            "rv_q": r.get("rv_q"), "rv_qa": r.get("rv_qa"), "n": r.get("n"),
        })
        cell[(o, t, r["group"])] = {
            "cf_y": float(r["cf_y"]), "cf_d": float(r["cf_d"]),
            "dt": float(r["delta_theta"]),
        }

    # Data-driven panels: only render a panel whose treatments are actually
    # present.  The APOS sensitivity path has no any-treatment level, so Panel A
    # is dropped automatically (and Panel B becomes the sole, unnumbered panel).
    panelA = [("Any treatment", "water_treatment")] if "water_treatment" in treats else []
    panelB = [(lbl, key) for key, lbl in (
        ("treat_boil", "Boil"), ("treat_chlorine", "Chlorine"),
        ("treat_filter", "Filter"), ("treat_other", "Other"))
        if key in treats]
    if panelA:
        row_groups = [("Panel A: Any treatment", panelA),
                      ("Panel B: Specific treatments", panelB)]
    else:
        row_groups = [("Method vs.\\ no treatment", panelB)]

    def _mc(span, body, i):
        s = "c" if i == O - 1 else "c|"
        return rf"\multicolumn{{{span}}}{{{s}}}{{{body}}}"

    col_spec = "l|" + "|".join("ccc" for _ in outcomes)
    # Landscape height budget is \textwidth (rotated); portrait is \textheight.
    _hcap = r"0.72\textwidth" if landscape else r"0.88\textheight"
    out = [r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
           *( [r"\begin{landscape}"] if landscape else [] ),
           r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{caption or r'APOS: Confounder Benchmarking --- Observed Controls vs.\ the Robustness Value'}}}",
           rf"\label{{{label or 'tab:sensitivity-benchmark'}}}",
           r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\renewcommand{\arraystretch}{0.92}",
           # adjustbox scales to fit BOTH width and height (resizebox only caps
           # width, so a tall table overruns the page bottom).
           rf"\begin{{adjustbox}}{{max width=\linewidth, max totalheight={_hcap}, center}}",
           rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule"]

    # outcome super-header + cmidrules
    sup = [""] + [_mc(SUB, _get_outcome_label(o), i) for i, o in enumerate(outcomes)]
    out.append(" & ".join(sup) + r" \\")
    out.append(" ".join(
        rf"\cmidrule(lr){{{2 + i * SUB}-{1 + (i + 1) * SUB}}}" for i in range(O)))
    # sub-header: cf_y cf_d dtheta
    sub_h = [""]
    for _ in outcomes:
        sub_h += [r"cf$_y$", r"cf$_d$", r"$\Delta\theta$"]
    out.append(" & ".join(sub_h) + r" \\")
    out.append(r"\midrule")

    for gi, (panel_title, rows) in enumerate(row_groups):
        if not rows:
            continue
        out.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{panel_title}}}}} \\")
        for ti, (rlabel, tkey) in enumerate(rows):
            # Treatment header: ATE with RV / RV_alpha folded in, one spanning
            # cell per outcome (e.g. "-0.150*** [RV 11.3/10.3]").
            head = [rlabel]
            for i, o in enumerate(outcomes):
                sp = spec.get((o, tkey))
                if sp is None:
                    head.append(_mc(SUB, "---", i))
                    continue
                coef = _format_coef(sp["ate"], sp["se"])
                rv = sp["rv_q"]; rva = sp["rv_qa"]
                rvs = (rf" {{\scriptsize[RV {100*float(rv):.1f}/{100*float(rva):.1f}]}}"
                       if rv is not None and rv == rv else "")
                head.append(_mc(SUB, coef + rvs, i))
            out.append(" & ".join(head) + r" \\")

            # one row per confounder group: cf_y, cf_d, dtheta
            for gkey, glabel, _desc in _sens_groups():
                line = [rf"\quad {glabel}"]
                any_val = False
                for o in outcomes:
                    c = cell.get((o, tkey, gkey))
                    if c is None:
                        line += ["", "", ""]
                    else:
                        any_val = True
                        line += [f"{100*c['cf_y']:.1f}", f"{100*c['cf_d']:.1f}",
                                 f"{100*c['dt']:+.2f}"]
                if any_val:
                    out.append(" & ".join(line) + r" \\")
            if ti < len(rows) - 1:
                out.append(r"\midrule")
        if gi < len(row_groups) - 1:
            out.append(r"\midrule")

    # Short legend only (the verbose glossary / design / N paragraph was removed).
    notes = (
        r"\textit{Notes:} cf$_y$, cf$_d$ = partial $R^2$ ($\times100$) of each "
        r"confounder group in the outcome and in adoption; $\Delta\theta$ "
        r"($\times100$) = bias from an unobserved confounder as strong as that group; "
        r"[RV\,/\,RV$_\alpha$] next to each ATE = robustness value (point estimate / "
        r"95\% CI). $^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.")

    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\footnotesize {notes}\end{{minipage}}",
            r"\end{table}",
            *( [r"\end{landscape}"] if landscape else [] )]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Table saved to: {filepath}")
    return filepath


# =============================================================================
# Combined grouped tables (outcomes x learners), coef-over-(se), panel rows
# =============================================================================
# LaTeX preamble required by these tables:
#   \usepackage{booktabs}    % \toprule \midrule \cmidrule \bottomrule
#   \usepackage{pdflscape}   % landscape
#   \usepackage{graphicx}    % \resizebox

_OUTCOME_ORDER = ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]
_DATASET_OF_OUTCOME = {"SomeRiskHome": "HH", "VeryHighRiskHome": "HH", "diarrhea": "U5"}
_CLUSTER_OF_DATASET = {"HH": "---", "U5": "HHID"}

# Base learners that compose the stacked ensemble (order as in _learners.py)
_STACK_BASE = ["ols", "lasso", "ridge", "enet", "rf", "xgb"]


def _fmt_w(w):
    """Format a stacking weight (0-1) or '' if missing."""
    if w is None or (isinstance(w, float) and w != w):
        return ""
    return f"{w:.2f}"


def _render_grouped_table(results, row_groups, caption, label, notes,
                          learner_keys=None, filename="table.tex", show_weights=True,
                          landscape=True):
    """Render a combined (outcome x learner) table with coef over (se) cells.

    row_groups : list of (panel_title_or_None, [(row_label, treatment_key), ...]).
    A cell is looked up by (outcome, treatment_key, learner); missing -> '---'.
    When show_weights and a stacked spec carries 'weights_g'/'weights_m', extra
    rows report each base learner's meta-weight (g | m) under the stacked column.
    landscape : wrap in a rotated landscape page (wide tables) when True; when
    False render upright (portrait) — only sensible for few-column tables.
    """
    learner_keys = learner_keys or list(LEARNER_NAMES)
    present = {r["outcome"] for r in results}
    outcomes = [o for o in _OUTCOME_ORDER if o in present]
    if not outcomes:
        outcomes = sorted(present)
    L, O = len(learner_keys), len(outcomes)
    ncol = 1 + O * L
    has_stacked = "stacked" in learner_keys

    idx, ymean, treat_share, nclust = {}, {}, {}, {}
    _apos_treated, _apos_total = {}, {}
    for r in results:
        o = r["outcome"]
        # Each outcome belongs to one dataset (E.coli -> HH, diarrhea -> U5).
        # Skip rows from the other dataset so a shared column name (e.g.
        # SomeRiskHome present in both files) does not overwrite the right one.
        want_ds = _DATASET_OF_OUTCOME.get(o)
        if want_ds and r.get("dataset_type") and r["dataset_type"] != want_ds:
            continue
        idx[(o, r["treatment"], r["learner"])] = r
        if r.get("y_mean") is not None:
            ymean.setdefault(o, float(r["y_mean"]))
        if r.get("n_clusters") is not None:
            nclust.setdefault(o, int(r["n_clusters"]))
        n = r.get("n")
        if r.get("method") == "APOS":
            # treated = any non-reference level; one row per level, so sum (per
            # learner the levels repeat -> de-dup on (outcome, treatment))
            key = (o, r.get("treatment"))
            if key not in _apos_total or True:
                _apos_total[o] = n
            _apos_treated.setdefault(key, r.get("n_treated") or 0)
        elif r.get("treatment") == "water_treatment" and n and r.get("n_treated") is not None:
            treat_share.setdefault(o, r["n_treated"] / n)

    # APOS treated proportion: sum of treated across treatment levels / N
    for o, tot in _apos_total.items():
        if tot:
            treated = sum(v for (oo, _t), v in _apos_treated.items() if oo == o)
            treat_share[o] = treated / tot

    # Vertical rule between outcome (dependent-variable) groups; the per-group
    # multicolumn cells must repeat that border, hence _mc().
    def _mc(span, body, i):
        spec = "c" if i == O - 1 else "c|"
        return rf"\multicolumn{{{span}}}{{{spec}}}{{{body}}}"

    col_spec = "l|" + "|".join("c" * L for _ in outcomes)
    # In landscape the usable page height is \textwidth (rotated); in portrait it
    # is \textheight.  Leave room for the caption + notes that sit OUTSIDE the box.
    _hcap = r"0.70\textwidth" if landscape else r"0.88\textheight"
    out = [r"% Requires: \usepackage{booktabs, pdflscape, graphicx, adjustbox}",
           *( [r"\begin{landscape}"] if landscape else [] ),
           r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           # \scriptsize + tight row spacing keeps the ~40-row body on one page;
           # \footnotesize overran by ~80pt.
           r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\renewcommand{\arraystretch}{0.92}",
           # adjustbox caps BOTH width and height: a wide table shrinks to fit
           # the line, a narrow one is NOT stretched (so it never overruns).
           rf"\begin{{adjustbox}}{{max width=\linewidth, max totalheight={_hcap}, center}}",
           rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule"]

    # outcome super-header + cmidrules
    sup = [""] + [_mc(L, _get_outcome_label(o), i) for i, o in enumerate(outcomes)]
    out.append(" & ".join(sup) + r" \\")
    out.append(" ".join(
        rf"\cmidrule(lr){{{2 + i * L}-{1 + (i + 1) * L}}}" for i in range(O)))
    # learner sub-header
    sub = [""] + [LEARNER_LABELS.get(lk, lk) for _ in outcomes for lk in learner_keys]
    out.append(" & ".join(sub) + r" \\")
    out.append(r"\midrule")

    # body
    for gi, (panel_title, rows) in enumerate(row_groups):
        if panel_title:
            out.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{panel_title}}}}} \\")
        for ti, (rlabel, tkey) in enumerate(rows):
            coef = [rlabel]
            se = [""]
            for o in outcomes:
                for lk in learner_keys:
                    r = idx.get((o, tkey, lk))
                    if r is None:
                        coef.append("---")
                        se.append("")
                    else:
                        coef.append(_format_coef(r["coef"], r["se"], r.get("ci_lower"), r.get("ci_upper")))
                        se.append(_format_se(r["se"]))
            out.append(" & ".join(coef) + r" \\")
            out.append(" & ".join(se) + r" \\")

            # Row order per treatment: coef, (se), w_g, w_m, RV, RV_alpha, N.

            # Stacking-weight rows (g then m): each base learner's meta-weight in
            # the stacked ensemble, shown under that learner's own column; blank
            # for non-base learners and for the stacked column itself.
            if show_weights and has_stacked:
                for nuis, wkey in (("g", "weights_g"), ("m", "weights_m")):
                    wrow = [rf"\quad $w_{nuis}$"]
                    any_w = False
                    for o in outcomes:
                        sr = idx.get((o, tkey, "stacked"))
                        wd = (sr or {}).get(wkey) or {}
                        for lk in learner_keys:
                            val = wd.get(lk)
                            if val is not None:
                                wrow.append(_fmt_w(val))
                                any_w = True
                            else:
                                wrow.append("")
                    if any_w:
                        out.append(" & ".join(wrow) + r" \\")

            # Sensitivity rows: RV then RV_alpha (Chernozhukov et al. 2022) for
            # that exact spec (outcome x treatment x learner).
            for rvkey, rvlabel in (("rv_q", r"\quad RV"), ("rv_qa", r"\quad RV$_\alpha$")):
                row = [rvlabel]
                any_rv = False
                for o in outcomes:
                    for lk in learner_keys:
                        rr = idx.get((o, tkey, lk))
                        val = rr.get(rvkey) if rr else None
                        if val is not None:
                            row.append(f"{val:.2f}")
                            any_rv = True
                        else:
                            row.append("")
                if any_rv:
                    out.append(" & ".join(row) + r" \\")

            # Per-treatment N (centered under each outcome). Single-method
            # subsamples shrink N for the specific-treatment rows.
            nline = [r"\quad $N$"]
            for i, o in enumerate(outcomes):
                rr = next((idx.get((o, tkey, lk)) for lk in learner_keys
                           if idx.get((o, tkey, lk)) is not None), None)
                nval = rr.get("n") if rr else None
                nline.append(_mc(L, f"{nval:,}" if nval else "---", i))
            out.append(" & ".join(nline) + r" \\")

            # horizontal rule between treatment blocks within a panel
            if ti < len(rows) - 1:
                out.append(r"\midrule")
        # heavier division between panels
        if gi < len(row_groups) - 1:
            out.append(r"\midrule")

    # Footer: outcome mean, treated proportion, clusters (per outcome). N is
    # per treatment (body); cross-fit / trimming live in the Notes.
    out.append(r"\midrule")

    def _foot(label, fmt, have):
        cells = [label]
        for i, o in enumerate(outcomes):
            v = have.get(o)
            cells.append(_mc(L, fmt(v) if v is not None else "---", i))
        return " & ".join(cells) + r" \\"

    out.append(_foot(r"Outcome mean $\bar Y$", lambda v: f"{v:.3f}", ymean))
    out.append(_foot("Treated (\\%)", lambda v: f"{100 * v:.1f}", treat_share))
    out.append(_foot("Clusters", lambda v: f"{v:,}", nclust))

    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\vspace{3pt}",
            rf"\begin{{minipage}}{{\linewidth}}\scriptsize {notes}\end{{minipage}}",
            r"\end{table}",
            *( [r"\end{landscape}"] if landscape else [] )]

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Table saved to: {filepath}")
    return filepath


_CONFOUNDER_NOTE = (
    r"Controls (all models): household wealth quintile, education of the household head, "
    r"country fixed effects, urban/rural residence, main drinking-water source, an indicator "
    r"for any under-5 child in the household, indicators for a girl and a boy under~15, "
    r"sanitation-facility type, and deciles of the source-water E.coli test (100\,ml). "
    r"The diarrhea (under-5) model additionally controls for the child's age and sex. ")


_SPECIFIC_ROWS = [
    ("Boil", "treat_boil"), ("Chlorine", "treat_chlorine"),
    ("Filter", "treat_filter"), ("Other", "treat_other")]

_IRM_RV_NOTE = (
    r"The $w_g$ and $w_m$ rows give each base learner's meta-weight in the stacked "
    r"ensemble (outcome model $g$ and propensity $m$), shown under that learner's own "
    r"column; blank for non-base learners and for the stacked column itself. "
    r"The RV and RV$_\alpha$ rows are the Chernozhukov et al. (2022) robustness value (and its "
    r"CI-adjusted version): the share of residual variation an unobserved confounder must "
    r"explain in both $D$ and $Y$ to move the point estimate (its 95\% CI) to zero. " + _CONFOUNDER_NOTE +
    rf"Cross-fitting: {N_FOLDS} folds $\times$ {N_REP} reps; propensity trimmed to $[0.01,0.99]$. "
    r"Cluster: HHID for diarrhea (U5); E.coli outcomes estimated i.i.d. "
    r"$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.")


# Note shared by the combined results tables (E.coli, diarrhea): explains the
# IRM top row and the APOS contrast rows below it.
_RESULTS_NOTE = (
    r"\textit{Notes:} Each cell reports a coefficient with significance stars over its "
    r"(standard error); columns group the outcome(s), and within each the ML nuisance "
    r"learners. The top row, \emph{Any treatment}, is the DDML interactive-regression (IRM) "
    r"ATE of any household water treatment vs.\ none (full sample, ATE score). The remaining "
    r"rows are Average Potential Outcome (AIPW) contrasts, ATE($d$)$=$E[$Y(d)$]$-$E[$Y(0)$], "
    r"for each specific method vs.\ no treatment, estimated jointly on the full sample with a "
    r"multi-class propensity P($D{=}d\mid X$). The $w_g$ and $w_m$ rows give each base "
    r"learner's meta-weight in the stacked ensemble (outcome model $g$, propensity $m$), "
    r"shown under that learner's own column. The RV and RV$_\alpha$ rows are the Chernozhukov "
    r"et al.\ (2022) robustness value and its CI-adjusted version. " + _CONFOUNDER_NOTE +
    rf"Cross-fitting: {N_FOLDS} folds $\times$ {N_REP} reps; propensity trimmed to $[0.01,0.99]$. "
    r"IRM clusters on HHID for diarrhea (U5) and is i.i.d.\ for the E.coli outcomes; APOS SEs "
    r"come from \texttt{causal\_contrast} (i.i.d.; DoubleMLAPOS has no clustered SE). "
    r"$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.1.")


def create_results_table(results, outcomes, filename, caption, label,
                         learner_keys=None, notes=None, landscape=True):
    """Combined results table for a set of outcomes: IRM(any) on top, APOS below.

    Restricts ``results`` to ``outcomes`` (e.g. the two E.coli outcomes, or
    diarrhea alone), then renders a single panel whose top row is the
    any-treatment IRM ATE and whose remaining rows are the APOS method-vs-none
    contrasts (Boil / Chlorine / Filter / Other).  Used to split the headline
    results into a standalone E.coli table and a standalone diarrhea table.
    """
    keep = set(outcomes)
    sub = [r for r in results if r.get("outcome") in keep]
    irm_any = [r for r in sub
               if r.get("method") != "APOS"
               and r.get("treatment") == "water_treatment"
               and r.get("subgroup_var") is None]
    apos = [r for r in sub if r.get("method") == "APOS"]
    combined = irm_any + apos
    row_groups = [(None, [
        ("Any treatment", "water_treatment"),
        ("Boil", "Boil"), ("Chlorine", "Chlorine"),
        ("Filter", "Filter"), ("Other", "Other")])]
    return _render_grouped_table(
        combined, row_groups, caption, label,
        notes if notes is not None else _RESULTS_NOTE,
        learner_keys, filename, landscape=landscape)
