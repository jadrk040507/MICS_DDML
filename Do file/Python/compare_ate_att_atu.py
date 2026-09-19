"""Joint ATE/ATT/derived-ATU decomposition from trusted project checkpoints.

Run: .venv/bin/python compare_ate_att_atu.py [--selected-countries]
No fitting. Main tables aggregate all cross-fitting repetitions; only aggregated tables are published:
medians do not commute with linear combinations. Clustered SEs use PSU sums; ordinary-fold SEs use individual scores.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

# Old project checkpoints reference these two classes in __main__.
from ate import ConvexRegressor, ConvexClassifier, ANALYSIS_SPECS, PROJECT


def joint_effects(a, t, influence_a, influence_t, treated, clusters):
    """Delta-method joint inference for (ATE, ATT, ATU, p), one repetition.

    ATU=(ATE-p*ATT)/(1-p). Its influence includes estimated p and all
    covariances. Treatment effects retain the treated-minus-untreated sign.
    """
    d = np.asarray(treated, dtype=float)
    ia, it = np.asarray(influence_a), np.asarray(influence_t)
    if d.ndim != 1 or not np.isin(d, [0, 1]).all():
        raise ValueError('Treatment must be a binary vector')
    if ia.shape != d.shape or it.shape != d.shape or len(clusters) != len(d):
        raise ValueError('Influences, treatment, and clusters must align')
    if not np.isfinite(np.r_[a, t, ia, it]).all() or pd.isna(clusters).any():
        raise ValueError('Nonfinite inputs or missing clusters')
    p = d.mean()
    if not 0 < p < 1:
        raise ValueError('Both target populations are required')
    u = (a - p * t) / (1 - p)
    ip = d - p
    iu = (ia - p * it + (u - t) * ip) / (1 - p)
    influence = np.column_stack([ia, it, iu, ip])
    codes, labels = pd.factorize(clusters)
    if len(labels) < 2:
        raise ValueError('At least two PSUs required')
    sums = np.zeros((len(labels), 4))
    np.add.at(sums, codes, influence)
    covariance = sums.T @ sums / len(d)**2
    return np.array([a, t, u, p]), covariance


def linear_combination(theta, covariance, weights):
    """Fixed-coefficient contrast with its full joint covariance."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or not np.isfinite(w).all():
        raise ValueError('Provide four finite weights: ATE, ATT, ATU, p')
    return float(w @ theta), float(np.sqrt(max(0., w @ covariance @ w)))


def load_pair(dataset, outcome, method, prefix):
    name = f'{prefix}{dataset}_{outcome}_{method}.pkl'
    paths = [PROJECT / 'Output' / 'ATE' / 'checkpoints' / name,
             PROJECT / 'Output' / 'ATT' / 'checkpoints' / name]
    models = []
    for path in paths:
        m = joblib.load(path)
        models.append(m['model'] if isinstance(m, dict) else m)
    left, right = models
    pd.testing.assert_frame_equal(left._dml_data.data, right._dml_data.data,
                                  check_exact=True)
    if method.startswith('IRM') and (left.score != 'ATE' or right.score != 'ATTE'):
        raise ValueError('Unexpected source scores')
    frameworks = ([m.framework for m in models] if method.startswith('IRM')
                  else [m.causal_contrast(reference_levels=[0]) for m in models])
    if frameworks[0].all_thetas.shape != frameworks[1].all_thetas.shape:
        raise ValueError('Incompatible repetitions/contrasts')
    if frameworks[0].treatment_names != frameworks[1].treatment_names:
        raise ValueError('Contrast labels differ')
    return models, frameworks, paths


def run_scope(selected_countries=False):
    prefix = 'selected_countries_' if selected_countries else ''
    suffix = '_selected_countries' if selected_countries else ''
    # Each scope reads both clustered and ordinary-fold checkpoints.
    destination = PROJECT / 'Output' / 'ATE_ATT_ATU'
    destination.mkdir(parents=True, exist_ok=True)
    rows, joint_rows, weight_rows = [], [], []
    for dataset, _, _, outcome in ANALYSIS_SPECS:
        for method in ('IRM_clustered', 'APOS', 'IRM_iid', 'APOS_iid'):
            models, (fa, ft), paths = load_pair(dataset, outcome, method, prefix)
            data = models[0]._dml_data.data
            d = (data['water_treatment'].to_numpy() if method.startswith('IRM')
                 else data['WQ15_g'].ne(0).to_numpy())
            for j, contrast in enumerate(fa.treatment_names):
                for r in range(fa.all_thetas.shape[1]):
                    theta, cov = joint_effects(
                        fa.all_thetas[j, r], ft.all_thetas[j, r],
                        fa.scaled_psi[:, j, r], ft.scaled_psi[:, j, r],
                        d, data['Cluster_var'].to_numpy() if not method.endswith('_iid') else np.arange(len(d)))
                    a, t, u, p = theta
                    meta = dict(dataset=dataset, outcome=outcome, method=method,
                                contrast=contrast, repetition=r + 1, n=len(d),
                                n_treated=int(np.sum(d)), n_untreated=int(len(d)-np.sum(d)),
                                clusters=data['Cluster_var'].nunique() if not method.endswith('_iid') else None, p=p,
                                source_ATE=str(paths[0]), source_ATT=str(paths[1]),
                                y_mean_untreated=float(data.loc[np.asarray(d) == 0, outcome].mean()),
                                treatment_shares={int(k): float(v) for k, v in
                                    data['WQ15_g' if not method.startswith('IRM') else 'water_treatment'].value_counts(normalize=True).items()})
                    joint_rows.append(dict(**meta, theta=theta, covariance=cov,
                                           order=['ATE', 'ATT', 'ATU', 'p']))
                    targets = [('ATE', [1,0,0,0]), ('ATT', [0,1,0,0]),
                               ('ATU_derived', [0,0,1,0]),
                               ('ATT_minus_ATU', [0,1,-1,0])]
                    for label, w in targets:
                        coef, se = linear_combination(theta, cov, w)
                        pval = 2 * norm.sf(abs(coef / se)) if se > 0 else float(coef == 0)
                        rows.append(dict(**meta, estimand=label, coef=coef, se=se,
                                         pval=pval, ci_lower=coef-norm.ppf(.975)*se,
                                         ci_upper=coef+norm.ppf(.975)*se,
                                         p_ATT=p*t, one_minus_p_ATU=(1-p)*u,
                                         reconstructed_ATE=p*t+(1-p)*u,
                                         identity_residual=a-p*t-(1-p)*u))
            if method.startswith('IRM'):
                for estimand, m in zip(('ATE', 'ATT'), models):
                    for nuisance, weights in m.convex_weights.items():
                        if not np.isclose(sum(weights.values()), 1):
                            raise ValueError('Super Learner weights do not sum to one')
                        for learner, weight in weights.items():
                            weight_rows.append(dict(dataset=dataset, outcome=outcome,
                                                    source_fit=estimand, method=method, nuisance=nuisance,
                                                    learner=learner, weight=weight))
            print(f'Completed {dataset}: {outcome}, {method}', flush=True)
    results = pd.DataFrame(rows)
    results.to_pickle(destination / f'results_ate_att_atu{suffix}.pkl')
    pd.DataFrame(joint_rows).to_pickle(destination / f'joint_covariance{suffix}.pkl')
    pd.DataFrame(weight_rows).to_pickle(destination / f'super_learner_weights{suffix}.pkl')
    pd.DataFrame(weight_rows).to_latex(destination / f'table_super_learner_weights{suffix}.tex',
                                      index=False, float_format='%.6f', escape=True, longtable=True)
    write_readable_tables(results, destination, suffix)
    print(destination)


OUTCOME_LABELS = {
    'SomeRiskHome': 'Some risk',
    'VeryHighRiskHome': 'Very high risk',
    'diarrhea': 'Diarrhea (U5)',
}
TREATMENT_LABELS = {
    'water_treatment': 'Any treatment',
    '1 vs 0': 'Boiling', '2 vs 0': 'Chlorination/tablets',
    '3 vs 0': 'Straining/settling', '98 vs 0': 'Other treatment',
}
ESTIMANDS = ['ATE', 'ATT', 'ATU_derived']


def effect_cell(row):
    """Match the original three-decimal coefficients and significance stars."""
    stars = '***' if row.pval < .01 else '**' if row.pval < .05 else '*' if row.pval < .1 else ''
    return f'{row.coef:.3f}{stars}'


def table_tex(caption, label, headers, rows, notes, columns):
    """Original booktabs/adjustbox style, including compact descriptive panels."""
    return '\n'.join([
        r'% Requires: \usepackage{booktabs,adjustbox}',
        r'\begin{table}[htbp]', r'\centering', r'\caption{' + caption + '}',
        r'\label{tab:' + label.replace('_', '-') + '}', r'\scriptsize',
        r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.92}',
        r'\begin{adjustbox}{max width=\linewidth}',
        r'\begin{tabular}{l' + 'c'*columns + '}', r'\hline\hline',
        *headers, *([r'\addlinespace[3pt]'] if columns == 9 else [r'\midrule']), *rows, r'\hline\hline', r'\end{tabular}',
        r'\end{adjustbox}', r'\par\vspace{3pt}',
        r'\begin{minipage}{\linewidth}\scriptsize \textit{Notes:} ' + notes +
        r'\end{minipage}', r'\end{table}', ''])


def aggregate_repetitions(results):
    """Use the project's repetition-wise aggregation after deriving each effect."""
    keys = ['dataset', 'outcome', 'method', 'contrast', 'estimand']
    expected = set(results.repetition.unique())
    rows = []
    for _, group in results.groupby(keys, sort=False):
        if set(group.repetition) != expected or group.repetition.duplicated().any():
            raise ValueError('Missing or duplicate repetitions')
        if group['p'].nunique() != 1 or group['n'].nunique() != 1:
            raise ValueError('Estimation sample differs across repetitions')
        row = group.iloc[0].to_dict()
        coef = float(group.coef.median())
        row.update(coef=coef,
                   se=float(((group.coef + 1.96*group.se).median()-coef)/1.96),
                   pval=float(group.pval.median()),
                   ci_lower=float(group.ci_lower.median()),
                   ci_upper=float(group.ci_upper.median()),
                   n_repetitions=len(group), repetition=0,
                   aggregation='project_median')
        # Median coefficients need not satisfy the decomposition. Never carry
        # a repetition's contributions into the aggregate as if they did.
        for field in ['p_ATT','one_minus_p_ATU','reconstructed_ATE','identity_residual']:
            row[field] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_readable_tables(results, destination, suffix):
    """Publish only results aggregated over all cross-fitting repetitions."""
    aggregate = aggregate_repetitions(results)
    aggregate.to_pickle(destination / f'results_ate_att_atu_aggregated{suffix}.pkl')
    render_tables(aggregate, destination, suffix, aggregated=True)
    write_inference_tables(aggregate, destination, suffix)


def write_inference_tables(aggregate, destination, suffix):
    """Use short ordinary tables; no longtable package is required."""
    scope = 'Selected Countries' if suffix else 'Full Sample'
    tables = []
    for (outcome, method), block in aggregate.groupby(['outcome','method'], sort=False):
        model = 'IRM' if method.startswith('IRM') else 'APOS'
        specification = 'Unclustered' if method.endswith('_iid') else 'Clustered'
        display = block[['contrast','estimand','coef','se','pval','ci_lower','ci_upper']].copy()
        display['contrast'] = display['contrast'].map(TREATMENT_LABELS)
        display['estimand'] = display['estimand'].replace({'ATU_derived':'ATU', 'ATT_minus_ATU':'ATT-ATU'})
        display.columns = ['Treatment','Estimand','Coefficient','SE','p-value','CI lower','CI upper']
        caption = f'Treatment-Effect Inference -- {scope} -- {OUTCOME_LABELS[outcome]} -- {model} -- {specification}'
        tables += [r'% Requires: \usepackage{booktabs,adjustbox}',
                   r'\begin{table}[htbp]', r'\centering',
                   r'\caption{' + caption + '}',
                   r'\label{tab:inference-' + f'{outcome}-{method}{suffix}'.replace('_','-') + '}',
                   r'\scriptsize', r'\begin{adjustbox}{max width=\linewidth}',
                   display.to_latex(index=False,float_format='%.6f',escape=True,longtable=False),
                   r'\end{adjustbox}', r'\par\smallskip',
                   r'\begin{minipage}{\linewidth}\scriptsize '
                   r'\textit{Notes:} Probability units. Estimates aggregate all cross-fitting repetitions. '
                   r'ATU is derived; SE denotes standard error. CI endpoints are pointwise 95\% '
                   r'confidence limits. Inference includes joint covariance and estimated treatment shares. '
                   r'\end{minipage}', r'\end{table}', '']
    (destination / f'table_inference{suffix}.tex').write_text('\n'.join(tables),encoding='utf-8')


def render_tables(results, destination, suffix, aggregated=False):
    """Replicate the ATE/ATT publication layout without aggregating repetitions."""
    scope = 'selected countries' if suffix else 'full sample'
    outcomes = list(OUTCOME_LABELS)
    for iid in (False, True):
        tag = '_iid' if iid else '_clustered'
        spec = 'ordinary folds' if iid else 'clustered folds'
        subset = results.loc[results.method.str.endswith('_iid').eq(iid)]
        md = [f'# ATE, ATT and derived ATU: {scope}, {spec}', '',
              'Coefficients are in probability units: **0.01 = one percentage point**. '
              'Standard errors are in parentheses. *** p<0.01, ** p<0.05, * p<0.1.', '',
              'ATE averages over the analysis sample; ATT targets users of any treatment; '
              'ATU targets nonusers. The same target populations apply to each APOS contrast. '
              'ATU is derived, not separately fitted. ' +
              ('Main results aggregate all cross-fitting repetitions using the project median rule. '
               'ATT-minus-ATU is derived in each repetition and then aggregated; it need not equal the difference of displayed medians. '
               'Median coefficients need not satisfy the exact ATE decomposition.' if aggregated else
               'Appendix: all individual cross-fitting repetitions.'), '']
        main_tables, contrast_tables, decomposition_tables = [], [], []
        for rep, block in subset.groupby('repetition', sort=True):
            rows, diff_rows, decomp_rows = [], [], []
            rep_label = f'aggregated over {int(block.n_repetitions.iloc[0])} repetitions' if aggregated else f'repetition {rep}'
            md += [f'## Cross-fitting: {rep_label}', '']
            for panel in ('IRM', 'APOS'):
                method = ('IRM_iid' if iid else 'IRM_clustered') if panel == 'IRM' else ('APOS_iid' if iid else 'APOS')
                contrasts = ['water_treatment'] if panel == 'IRM' else ['1 vs 0','2 vs 0','3 vs 0','98 vs 0']
                panel_data = block.loc[block.method.eq(method)]
                rows.append(r'\multicolumn{10}{l}{\textit{' + panel + r'}} \\')
                diff_rows.append(r'\multicolumn{4}{l}{\textit{' + panel + r'}} \\')
                decomp_rows.append(r'\multicolumn{10}{l}{\textit{' + panel + r'}} \\')
                md += [f'### {panel}', '', '| Treatment / outcome | ATE | ATT | ATU | ATT−ATU |',
                       '|---|---:|---:|---:|---:|']
                for contrast in contrasts:
                    cells, errors, differences, diff_errors, contributions = [], [], [], [], []
                    for outcome in outcomes:
                        values = panel_data.loc[panel_data.outcome.eq(outcome) & panel_data.contrast.eq(contrast)].set_index('estimand')
                        for e in ESTIMANDS:
                            v = values.loc[e]
                            cells.append(effect_cell(v)); errors.append(f'({v.se:.3f})')
                        delta = values.loc['ATT_minus_ATU']
                        differences.append(effect_cell(delta)); diff_errors.append(f'({delta.se:.3f})')
                        v = values.loc['ATE']
                        contributions += [f'{v.p_ATT:.3f}', f'{v.one_minus_p_ATU:.3f}', f'{v.reconstructed_ATE:.3f}']
                        md.append('| ' + TREATMENT_LABELS[contrast] + ' / ' + OUTCOME_LABELS[outcome] + ' | ' +
                                  ' | '.join(effect_cell(values.loc[e]) + f' ({values.loc[e].se:.3f})'
                                             for e in [*ESTIMANDS,'ATT_minus_ATU']) + ' |')
                    label = TREATMENT_LABELS[contrast]
                    rows += [label + ' & ' + ' & '.join(cells) + r' \\', ' & ' + ' & '.join(errors) + r' \\']
                    diff_rows += [label + ' & ' + ' & '.join(differences) + r' \\', ' & ' + ' & '.join(diff_errors) + r' \\']
                    decomp_rows.append(label + ' & ' + ' & '.join(contributions) + r' \\')
                rows += [r'\addlinespace[6pt]', r'\multicolumn{10}{l}{\textit{Descriptive statistics and sample}} \\']
                samples = [panel_data.loc[panel_data.outcome.eq(o)].iloc[0] for o in outcomes]
                descriptive = [('Y mean, no treatment (\\%)', [f'{100*v.y_mean_untreated:.1f}\\%' for v in samples]),
                               ('Treated (\\%)', [f'{100*v.p:.1f}\\%' for v in samples]),
                               ('Observations', [f'{v.n:,.0f}' for v in samples])]
                if not iid:
                    descriptive.append(('PSUs', [f'{v.clusters:,.0f}' for v in samples]))
                if panel == 'APOS':
                    for level, label in [(0,'No treatment'),(1,'Boiling'),(2,'Chlorination/tablets'),(3,'Straining/settling'),(98,'Other treatment')]:
                        descriptive.append((label + ' (\\%)', [f'{100*v.treatment_shares.get(level,0):.1f}\\%' for v in samples]))
                for label, vals in descriptive:
                    rows.append(label + ' & ' + ' & '.join(r'\multicolumn{3}{c}{' + v + '}' for v in vals) + r' \\')
                rows.append(r'\midrule'); diff_rows.append(r'\midrule'); decomp_rows.append(r'\midrule')
                md += ['', '| Outcome | N | Treated (%) | Y mean, no treatment (%) |', '|---|---:|---:|---:|']
                md += [f'| {OUTCOME_LABELS[o]} | {v.n:,.0f} | {100*v.p:.2f} | {100*v.y_mean_untreated:.2f} |'
                       for o,v in zip(outcomes,samples)]
                if not aggregated:
                    md += ['', '| Treatment / outcome | p × ATT | (1−p) × ATU | Sum = ATE |', '|---|---:|---:|---:|']
                    for _,v in panel_data.loc[panel_data.estimand.eq('ATE')].iterrows():
                        md.append(f'| {TREATMENT_LABELS[v.contrast]} / {OUTCOME_LABELS[v.outcome]} | {v.p_ATT:.3f} | {v.one_minus_p_ATU:.3f} | {v.reconstructed_ATE:.3f} |')
                    md += ['']
            header = ' & ' + ' & '.join(r'\multicolumn{3}{c}{' + v + '}' for v in OUTCOME_LABELS.values()) + r' \\'
            notes = (r'Coefficients are in probability units (0.01 equals one percentage point); standard errors are in parentheses. '
                     + ('Folds and inference are observation-level. ' if iid else 'Folds keep PSUs together; standard errors are PSU-cluster robust. ')
                     + r'ATE averages over the analysis sample; ATT targets users of any water treatment; ATU targets nonusers. '
                     r'For U5, targets refer to children in these households. All APOS contrasts use the same any-treatment target for ATT. '
                     r'ATU is derived as $(ATE-p\,ATT)/(1-p)$; inference includes joint covariance and estimated $p$. '
                     r'Descriptive statistics refer to the full estimation sample, without survey weights. '
                      + (r'Coefficients, p-values and interval endpoints are medians across repetitions. The SE is recovered from the median upper bound using the project rule. ATT minus ATU is aggregated after its repetition-wise calculation. Median coefficients need not satisfy the ATE decomposition. ' if aggregated else r'Each table reports one cross-fitting repetition, not the median across repetitions. ') +
                     r'$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$; pointwise tests without multiplicity adjustment.')
            notes = f'Sample: {scope}; {spec}; {rep_label}. ' + notes
            title_detail = 'Selected Countries' if suffix else 'Full Sample'
            if iid:
                title_detail += ' -- Unclustered'
            if not aggregated:
                title_detail += f' -- Repetition {rep}'
            caption = 'ATE, ATT and ATU -- ' + title_detail
            base = f'ate-att-atu{suffix}{tag}-' + ('aggregated' if aggregated else f'r{rep}')
            main_tables.append(table_tex(caption, base,
                                [header, r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}', ' & ' + ' & '.join(['ATE','ATT','ATU']*3) + r' \\'], rows[:-1], notes, 9))
            contrast_tables.append(table_tex('ATT--ATU Differences -- ' + title_detail,base+'-difference',
                                [' & ' + ' & '.join(OUTCOME_LABELS.values()) + r' \\'], diff_rows[:-1],
                                notes + ' Stars in this table test ATT minus ATU directly.', 3))
            decomposition_tables.append(table_tex('ATE Decomposition -- ' + title_detail,base+'-decomposition',
                                [header, r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}', ' & ' + ' & '.join([r'$p\,ATT$',r'$(1-p)\,ATU$','ATE']*3) + r' \\'],decomp_rows[:-1],
                                f'Sample: {scope}; {spec}; {rep_label}. ' + r'Probability units. Contributions sum to ATE before rounding. This identity holds by construction and is not an independent validation. '
                                r'$p$ is the treated share in the corresponding estimation sample.',9))
        stem = f'comparison{suffix}{tag}'  # Output filenames are in English.
        (destination / f'{stem}.md').write_text('\n'.join(md),encoding='utf-8')
        (destination / f'{stem}.tex').write_text('\n'.join(main_tables),encoding='utf-8')
        for label,tables in ([('contrasts',contrast_tables)] if aggregated else [('contrasts',contrast_tables),('decomposition',decomposition_tables)]):
            (destination / f'table_{label}{suffix}{tag}.tex').write_text('\n'.join(tables),encoding='utf-8')


def write_pdf(destination, selected_only=False):
    """Compile the English publication tables, followed by companion tables."""
    import re
    import shutil
    import subprocess
    import tempfile
    if shutil.which('pdflatex') is None:
        print('pdflatex is unavailable; Markdown and LaTeX tables are ready.')
        return
    parts = [r'\documentclass[10pt]{article}', r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}',
             r'\usepackage[a4paper,margin=1.5cm]{geometry}', r'\usepackage{booktabs,adjustbox,float,hyperref}',
             r'\setlength{\parindent}{0pt}', r'\begin{document}']
    scopes = ['_selected_countries'] if selected_only else ['', '_selected_countries']
    for family in ('main','contrasts'):
        for tag in ('_clustered','_iid'):
            for suffix in scopes:
                files = {
                    'main': [f'comparison{suffix}{tag}.tex'],
                    'contrasts': [f'table_contrasts{suffix}{tag}.tex'],
                }[family]
                extracted = [re.findall(r'\\begin\{table\}.*?\\end\{table\}',(destination/f).read_text(),re.S) for f in files]
                for rep in range(len(extracted[0])):
                    if len(parts)>6:
                        parts.append(r'\clearpage')
                    for tables in extracted:
                        parts.append(tables[rep].replace('[htbp]','[H]'))
    parts.append(r'\end{document}')
    name = 'comparison_ate_att_atu' + ('_selected_countries' if selected_only else '')
    tex = destination/f'{name}.tex'
    tex.write_text('\n'.join(parts),encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='mics-tables-') as temp:
        result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={temp}',str(tex.resolve())],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:])
        shutil.copy2(f'{temp}/{name}.pdf',destination/f'{name}.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-countries', action='store_true',
                        help='Only selected countries; default runs both scopes.')
    args = parser.parse_args()
    for selected in ([True] if args.selected_countries else [False, True]):
        run_scope(selected)
    write_pdf(PROJECT / "Output" / "ATE_ATT_ATU", args.selected_countries)


if __name__ == '__main__':
    main()
