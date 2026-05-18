import pickle
import pandas as pd
from pathlib import Path

cp_dir = Path('Output/checkpoints')
results = []
for f in sorted(cp_dir.glob('*.pkl')):
    with open(f, 'rb') as fh:
        r = pickle.load(fh)
    results.append(r)

df = pd.DataFrame(results)
print(f'PARTIAL RESULTS ({len(df)} estimates)')
print()

for (outcome, treatment), grp in df.groupby(['outcome','treatment']):
    stacked_row = grp[grp['learner']=='stacked']
    if len(stacked_row) > 0:
        stacked_coef = stacked_row['coef'].values[0]
        other_coefs = grp[grp['learner']!='stacked']['coef'].values
        mean_other = other_coefs.mean()
        diff = stacked_coef - mean_other
        print(f'{outcome:22s} | {treatment:20s} | stacked={stacked_coef:.4f} mean_others={mean_other:.4f} diff={diff:.4f}')

print()
for (outcome, treatment), grp in df.groupby(['outcome','treatment']):
    n_pos = (grp['coef'] > 0).sum()
    n_neg = (grp['coef'] < 0).sum()
    total = len(grp)
    consistent = 'YES' if n_pos == total or n_neg == total else 'MIXED'
    print(f'{outcome:22s} | {treatment:20s} | pos={n_pos} neg={n_neg}/{total} | {consistent}')

print()
print('RV VALUES:')
has_rv = df[df['rv_q'].notna()]
print(f'Rows with RV: {len(has_rv)} / {len(df)}')
for _, row in has_rv.iterrows():
    print(f'  {row["outcome"]} | {row["treatment"]} | {row["learner"]} | rv_q={row["rv_q"]:.4f} rv_qa={row["rv_qa"]:.4f}')