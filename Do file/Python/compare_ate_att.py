"""Compare the saved original (ATE) and ATT estimates without fitting models.

Run with the project's Python environment:
    python compare_ate_att.py
    python compare_ate_att.py --selected-countries

Inputs: Output/ATE/results_*.pkl and Output/ATT equivalents.
Outputs: Output/ATE_vs_ATT/table_ate_att_*.tex (clustered only),
         results_ate_att*.pkl (paired numeric results with source paths).
Default run: main effects, GATE, sensitivity, and IRM Super Learner weights;
selected-country effects and weights are also generated. Selected-country
GATE/sensitivity and APOS weights are not present in the saved source files.
Only pandas is required. Read result pickles produced by this project.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
KEYS = ['dataset', 'outcome', 'method', 'specification', 'index']
VALUES = ['coef', 'std err', 'P>|t|', '2.5 %', '97.5 %', 'n', 'clusters']
OUTCOMES = [('HH', 'SomeRiskHome', 'Some risk'),
            ('HH', 'VeryHighRiskHome', 'Very high risk'),
            ('U5', 'diarrhea', 'Diarrhea (U5)')]
TREATMENTS = [('IRM', 'water_treatment', 'Any treatment'),
              ('APOS', '1 vs 0', 'Boiling'),
              ('APOS', '2 vs 0', 'Chlorination/tablets'),
              ('APOS', '3 vs 0', 'Straining/settling'),
              ('APOS', '98 vs 0', 'Other treatment')]


def load_results(directory, estimand, suffix):
    """Validate source keys before pairing estimates; never join by row order."""
    frames = []
    for method in ('irm', 'apos'):
        path = directory / f'results_{method}{suffix}.pkl'
        frame = pd.read_pickle(path).copy()
        missing = set(KEYS + VALUES) - set(frame.columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        if 'estimand' in frame and not frame['estimand'].eq(estimand).all():
            raise ValueError(f'{path}: unexpected estimand; expected {estimand}')
        if not frame['method'].eq(method.upper()).all():
            raise ValueError(f'{path}: unexpected method')
        # Select the actual clustered label used by each original producer.
        clustered_label = 'clustered' if method == 'irm' else 'clustered_folds'
        frame = frame.loc[frame['specification'].eq(clustered_label)].copy()
        if frame.empty:
            raise ValueError(f'{path}: no {clustered_label} estimates')
        frame['specification'] = 'clustered'
        frame['source'] = str(path.resolve())
        frames.append(frame[KEYS + VALUES + ['source']])
    result = pd.concat(frames, ignore_index=True)
    if result[KEYS].isna().any().any() or result.duplicated(KEYS).any():
        raise ValueError(f'{directory}: missing or duplicate comparison keys')
    return result


def compare(original, att):
    """Require both versions for every row and retain each version's statistics."""
    if any(not frame['specification'].eq('clustered').all() for frame in (original, att)):
        raise ValueError('Only clustered estimates are allowed')
    result = original.merge(att, on=KEYS, how='outer', validate='one_to_one',
                            suffixes=('_ATE', '_ATT'), indicator=True)
    unmatched = result.loc[result['_merge'].ne('both'), KEYS + ['_merge']]
    if not unmatched.empty:
        raise ValueError('Unmatched ATE/ATT estimates:\n' + unmatched.to_string(index=False))
    result = result.drop(columns='_merge')
    # This is a descriptive difference, not a test of equality. Inference for
    # the difference would require the covariance between the two estimators.
    result['ATT_minus_ATE'] = result['coef_ATT'] - result['coef_ATE']
    return result


def coefficient(row, estimand):
    value, p = row[f'coef_{estimand}'], row[f'P>|t|_{estimand}']
    if pd.isna(value):
        return '---'
    stars = '' if pd.isna(p) else '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''
    return f'{value:.3f}{stars}'


def write_table(frame, path):
    """Write paired ATE/ATT columns for each of the three original outcomes."""
    specification = 'clustered'
    if not frame['specification'].eq(specification).all():
        raise ValueError('Only clustered estimates are allowed')
    data = frame.set_index(KEYS)
    lines = [r'% Requires: \usepackage{booktabs,adjustbox}',
             r'\begin{table}[htbp]', r'\centering',
             r'\caption{Original (ATE) and ATT water-treatment effects: clustered'
             + (' (selected countries)' if 'selected_countries' in path.stem else '') + '}',
             r'\label{tab:' + path.stem.replace('_', '-') + '}',
             r'\scriptsize', r'\begin{adjustbox}{max width=\linewidth}',
             r'\begin{tabular}{lrrrrrr}', r'\toprule',
             ' & ' + ' & '.join(r'\multicolumn{2}{c}{' + label + '}'
                                for _, _, label in OUTCOMES) + r' \\',
             ' & ' + ' & '.join(['ATE', 'ATT'] * 3) + r' \\', r'\midrule']
    for method in ('IRM', 'APOS'):
        lines.append(r'\multicolumn{7}{l}{\textit{' + method + r'}} \\')
        for row_method, contrast, label in TREATMENTS:
            if row_method != method or contrast == '98 vs 0':
                continue
            rows = [data.loc[(dataset, outcome, method, specification, contrast)]
                    for dataset, outcome, _ in OUTCOMES]
            cells = [coefficient(row, estimand) for row in rows for estimand in ('ATE', 'ATT')]
            errors = ['---' if pd.isna(row[f'std err_{e}']) else f"({row[f'std err_{e}']:.3f})"
                      for row in rows for e in ('ATE', 'ATT')]
            lines.extend([label + ' & ' + ' & '.join(cells) + r' \\',
                          ' & ' + ' & '.join(errors) + r' \\'])
        # Report sample sizes for each method/version, without assuming equality.
        contrast = 'water_treatment' if method == 'IRM' else '1 vs 0'
        rows = [data.loc[(ds, out, method, specification, contrast)] for ds, out, _ in OUTCOMES]
        for stat, label in [('n', 'Observations'), ('clusters', 'PSUs')]:
            cells = ['---' if pd.isna(row[f'{stat}_{e}']) else f"{row[f'{stat}_{e}']:,.0f}"
                     for row in rows for e in ('ATE', 'ATT')]
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.extend([r'\end{tabular}', r'\end{adjustbox}', r'\par\vspace{3pt}',
                  r'\begin{minipage}{\linewidth}\scriptsize '
                  r'\textit{Notes:} Coefficients and standard errors in parentheses are taken from saved results. '
                  r'Only clustered estimates are used: folds keep PSUs together and standard errors are PSU-cluster robust. '
                  r'ATE averages over the analysis population; ATT targets observations in households using any water treatment '
                  r'(including children in those households for the U5 outcome). '
                  r'All APOS contrasts compare the stated method with no treatment; ATT APOS uses the same '
                  r'any-treatment target population for every contrast. '
                  r'Observations report the full estimation sample, not the number of treated observations. '
                  r'Coefficients are in probability units (0.01 equals one percentage point). '
                  r'The two columns target different estimands; no test of their difference is reported. '
                  r'$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. '
                  r'\end{minipage}', r'\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def load_comparison(kind, suffix=''):
    """Pair additional result families, retaining all source statistics."""
    keys = {
        'heterogeneity_gates': ['dataset', 'outcome', 'method', 'group', 'group_value', 'treatment_label'],
        'sensitivity': ['dataset', 'outcome', 'method', 'treatment'],
        'convex_weights': ['dataset', 'outcome', 'model', 'nuisance', 'learner'],
    }[kind]
    frames = []
    for estimand, directory in [('ATE', PROJECT / 'Output' / 'ATE'), ('ATT', PROJECT / 'Output' / 'ATT')]:
        path = directory / f'results_{kind}{suffix}.pkl'
        frame = pd.read_pickle(path).copy()
        if 'estimand' in frame and not frame['estimand'].eq(estimand).all():
            raise ValueError(f'{path}: unexpected estimand')
        if kind == 'convex_weights':
            frame = frame.loc[frame['model'].str.endswith('_IRM_clustered')].copy()
            frame['specification'] = 'clustered'
        else:
            frame = frame.loc[frame['specification'].eq('clustered_folds')].copy()
            frame['specification'] = 'clustered'
        if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f'{path}: empty clustered data or invalid comparison keys')
        frame = frame.drop(columns='estimand', errors='ignore')
        frame['source'] = str(path.resolve())
        frames.append(frame)
    result = frames[0].merge(frames[1], on=keys, how='outer',
                            validate='one_to_one', suffixes=('_ATE', '_ATT'), indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError(f'{kind}{suffix}: unmatched ATE/ATT rows')
    if kind == 'heterogeneity_gates':
        for field in ['group_label', 'source_ecoli_range']:
            if not result[f'{field}_ATE'].equals(result[f'{field}_ATT']):
                raise ValueError(f'GATE groups differ between estimands: {field}')
    if kind == 'convex_weights':
        for estimand in ('ATE', 'ATT'):
            sums = result.groupby(['model', 'nuisance'])[f'weight_{estimand}'].sum()
            if not (sums - 1).abs().lt(1e-6).all():
                raise ValueError('Super Learner weights do not sum to one')
    return result.drop(columns='_merge')


def latex_text(value):
    """Escape plain source labels, including the top-coded E. coli range."""
    mapping = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%',
               '&': r'\&', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
               '>': r'\textgreater{}', '<': r'\textless{}'}
    return ''.join(mapping.get(char, char) for char in str(value))


def compact_table(path, caption, headers, rows, notes):
    """Render small tables; cell strings already contain escaped LaTeX."""
    lines = [r'% Requires: \usepackage{booktabs,adjustbox}', r'\begin{table}[htbp]',
             r'\centering\scriptsize', r'\caption{' + caption + '}',
             r'\label{tab:' + path.stem.replace('_', '-') + '}',
             r'\begin{adjustbox}{max width=\linewidth}',
             r'\begin{tabular}{' + 'l' * (len(headers) - 2) + 'rr}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}',
              r'\par\vspace{3pt}\begin{minipage}{\linewidth}\scriptsize\textit{Notes:} '
              + notes + r'\end{minipage}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_additional_tables(destination, kind, suffix=''):
    frame = load_comparison(kind, suffix)
    frame.to_pickle(destination / f'results_ate_att_{kind}{suffix}.pkl')
    scope = ' (selected countries)' if suffix else ''
    for dataset, outcome, label in OUTCOMES:
        subset = frame.loc[frame.dataset.eq(dataset) & frame.outcome.eq(outcome)]
        if subset.empty:
            raise ValueError(f'{kind}: missing {outcome}')
        rows = []
        if kind == 'heterogeneity_gates':
            headers = ['Treatment', 'Source group', 'CFU/100 mL', 'ATE', 'ATT']
            subset = subset.assign(_order=pd.to_numeric(subset.group_value)).sort_values(
                ['method', 'treatment_label', 'group', '_order'])
            for _, row in subset.iterrows():
                effects = []
                for e in ('ATE', 'ATT'):
                    values = {f'coef_{e}': row[f'coef_{e}'], f'P>|t|_{e}': row[f'pval_{e}']}
                    effects.append(coefficient(values, e) + f" ({row[f'se_{e}']:.3f})")
                rows.append([latex_text(row.treatment_label), latex_text(row['group_label_ATE']),
                             latex_text(row['source_ecoli_range_ATE'])] + effects)
            notes = (r'Clustered estimates by source-water E. coli group. Cells show coefficients '
                     r'and PSU-cluster-robust standard errors. ATE averages within the group; ATT '
                     r'targets observations in households using any treatment within that group, '
                     r'for all treatment contrasts. Only observed groups are shown; missing decile '
                     r'numbers are not imputed. Stars use saved pointwise p-values, not joint inference: '
                     r'$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$. '
                     r'Heterogeneity is exploratory. Stars do not test ATE versus ATT or differences '
                     r'between groups. Pointwise and joint confidence intervals are retained in the paired pickle.')
            title = 'GATE: original (ATE) versus ATT'
        elif kind == 'sensitivity':
            headers = ['Method', 'Treatment', 'Measure', 'ATE', 'ATT']
            labels = {'Any Treatment': 'Any treatment', '1': 'Boiling',
                      '2': 'Chlorination/tablets', '3': 'Straining/settling'}
            for _, row in subset.sort_values(['method', 'treatment']).iterrows():
                for metric, name in [('cf_y', r'$cf_y$'), ('cf_d', r'$cf_d$'),
                                     ('rv', 'RV'), ('rva', r'RV$_\alpha$')]:
                    cells = ['---' if pd.isna(row[f'{metric}_{e}']) else
                             f"{100 * row[f'{metric}_{e}']:.4f}\\%" for e in ('ATE', 'ATT')]
                    rows.append([row.method, latex_text(labels[str(row.treatment)]), name] + cells)
            notes = (r'Clustered results. All measures are percentages, with four decimals to preserve '
                     r'small values. The benchmark omits the source-water E. coli decile block. '
                     r'$cf_y$ and $cf_d$ are its benchmark confounding-strength measures. '
                     r'RV is the confounding strength required to bring the estimate to zero; '
                     r'RV$_\alpha$ brings its 95\% confidence interval to include zero. '
                     r'Comparisons retain the original sensitivity assumptions; ATE and ATT target '
                     r'different populations. Larger RV indicates greater robustness under those assumptions.')
            title = 'Sensitivity: original (ATE) versus ATT'
        else:
            headers = ['Nuisance', 'Learner', 'ATE fit', 'ATT fit']
            for _, row in subset.sort_values(['nuisance', 'learner']).iterrows():
                rows.append([latex_text(row.nuisance), latex_text(row.learner)] +
                            [f"{row[f'weight_{e}']:.6f}" for e in ('ATE', 'ATT')])
            notes = (r'Only clustered IRM weights are available in these result files. '
                     r'Weights are averaged across outer folds and repetitions. '
                     r'ml\_g0 and ml\_g1 predict outcomes under no treatment and treatment; '
                     r'ml\_m predicts treatment. The outcome components are reported separately, '
                     r'without pooling their weights. Each nuisance ensemble sums to one before rounding. '
                     r'These are prediction weights, not causal effects or variable importance. '
                     r'No APOS weights are inferred from IRM weights.')
            title = 'Super Learner weights: original versus ATT'
        path = destination / f'table_ate_att_{kind}_{outcome}{suffix}.tex'
        compact_table(path, title + ': ' + label + scope, headers, rows, notes)
        print(path)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-countries', action='store_true',
                        help='Generate only selected-country effects and weights.')
    args = parser.parse_args()
    destination = PROJECT / 'Output' / 'ATE_vs_ATT'
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = ('_selected_countries',) if args.selected_countries else ('', '_selected_countries')
    for suffix in suffixes:
        original = load_results(PROJECT / 'Output' / 'ATE', 'ATE', suffix)
        att = load_results(PROJECT / 'Output' / 'ATT', 'ATT', suffix)
        result = compare(original, att)
        path = destination / f'table_ate_att_main{suffix}.tex'
        write_table(result, path)
        print(path)
        result.to_pickle(destination / f'results_ate_att{suffix}.pkl')
        write_additional_tables(destination, 'convex_weights', suffix)
    if not args.selected_countries:
        for kind in ('heterogeneity_gates', 'sensitivity'):
            write_additional_tables(destination, kind)
    print('Done: clustered comparisons only. Selected-country GATE and sensitivity '
          'and APOS weights are not available in the saved source results.')


if __name__ == '__main__':
    main()
