import pickle
from pathlib import Path

for f in sorted(Path('Output/checkpoints').glob('*.pkl')):
    with open(f, 'rb') as fh:
        r = pickle.load(fh)
    rv_q = r.get('rv_q', 'N/A')
    rv_qa = r.get('rv_qa', 'N/A')
    if isinstance(rv_q, float):
        rv_q = f'{rv_q:.4f}'
    if isinstance(rv_qa, float):
        rv_qa = f'{rv_qa:.4f}'
    print(f'{f.name}: coef={r["coef"]:.4f} se={r["se"]:.4f} rv_q={rv_q} rv_qa={rv_qa}')