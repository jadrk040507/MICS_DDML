"""Generate ATE and ATT GATE tables for deciles AND three source-risk groups.

Uses saved GATE results when complete. Missing groupings are computed from saved
model scores, without refitting nuisance models. Run from any directory:
    .venv/bin/python generate_gate_tables.py
All tables report the existing aggregated cross-fitting inference.
"""
import gc
import joblib
import pandas as pd
import ate
import att
# Older checkpoints serialize learner classes under __main__.
from ate import ConvexRegressor, ConvexClassifier


class SavedModels:
    """Load one checkpoint at a time, using the existing bundle interface."""
    def __init__(self, module, dataset, outcome):
        self.module, self.prefix = module, f'{dataset}_{outcome}_'
        self.store = module.CheckpointStore(False)

    def __getitem__(self, key):
        suffix = {'irm_cluster':'IRM_clustered', 'irm_no_cluster':'IRM_iid',
                  'apos_cluster':'APOS', 'apos_no_cluster':'APOS_iid'}[key]
        value = joblib.load(self.store.path(f'{self.prefix}{suffix}'))
        return value['model'] if isinstance(value, dict) else value


def generate(module):
    path = module.OUTPUT_DIR / 'results_heterogeneity_gates.pkl'
    result = pd.read_pickle(path) if path.exists() else pd.DataFrame()
    ranges = module.source_ecoli_ranges_from_master_data()
    rows = [result] if not result.empty else []
    for dataset, data_path, child, outcome in module.ANALYSIS_SPECS:
        data = None
        for spec, clustered in [('clustered_folds',True), ('unclustered',False)]:
            for group in module.GATE_GROUPS:
                existing = result if result.empty else result.loc[
                    result.dataset.eq(dataset) & result.outcome.eq(outcome) &
                    result.specification.eq(spec) & result.group.eq(group)]
                if not existing.empty:
                    continue
                if data is None:
                    data = module.load_analysis_data(data_path,outcome,child,None,False)
                print(f'{module.__name__.upper()}: {outcome}, {spec}, {group}', flush=True)
                added = pd.concat(module.gate_for_group(
                    SavedModels(module,dataset,outcome),data,dataset,outcome,child,
                    spec,clustered,group),ignore_index=True)
                added['source_ecoli_range'] = [
                    module.SOURCE_RISK_RANGES[str(r.group_value)] if group=='source_risk'
                    else ranges[(dataset,str(r.group_value))] for r in added.itertuples()]
                rows.append(added)
        del data
        gc.collect()
    complete = pd.concat(rows,ignore_index=True)
    keys=['dataset','outcome','specification','group','group_value','method','treatment_label']
    if complete.duplicated(keys).any():
        raise ValueError('Duplicate GATE result keys')
    if len(rows)>1 or result.empty:
        complete.to_pickle(path)
    module.write_gate_tables(complete)
    print(f'{module.__name__.upper()}: {len(complete)} GATE rows; combined, decile and risk-group tables ready.',flush=True)


if __name__ == '__main__':
    for module in (ate,att):
        generate(module)
