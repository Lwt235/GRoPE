"""Matched place/seed results, including all unchanged strong old references."""
import json

import numpy as np

from experiments.gaussian_memory_visibility_training_data import (
    OUT, PARENT, KINDS, SEEDS, CONDITIONS, POLICIES, load_plan, sha256, write_json)
from experiments.gaussian_memory_visibility_data import FAMILIES as OLD_FAMILIES
from experiments.gaussian_memory_visibility_analysis import paired, delta_summary, METRICS

FAMILIES = list(OLD_FAMILIES) + ['new_content', 'new_soft', 'new_center']


def main():
    plan = load_plan(); catalog = sorted(plan['catalog'], key=lambda m: m['place'])
    places = [m['place'] for m in catalog]; records = {}; new_count = old_count = 0
    for m in catalog:
        path = PARENT / 'evaluation' / f'{m["id"]}.json'
        assert sha256(path) == plan['reused_record_sha256'][str(path)]
        old = json.loads(path.read_text()); new = json.loads((OUT / 'evaluation' / f'{m["id"]}.json').read_text())
        assert old['complete'] and new['complete']
        assert len(old['rows']) == 102 and len(new['rows']) == 54
        records[m['id']] = old['rows'] + new['rows']; new_count += 54; old_count += 102
    matrices = {}; conditions = {}; interactions = {}
    methods = ['native', 'local'] + [f'{f}/{p}' for f in FAMILIES for p in POLICIES]
    for ci, condition in enumerate(CONDITIONS):
        matrices[condition] = {}; conditions[condition] = {}
        for metric in METRICS:
            data = {}
            for method in methods:
                pieces = method.split('/'); family = pieces[0]; policy = pieces[1] if len(pieces) == 2 else None
                values = [[next(r[metric] for r in records[m['id']] if r['family'] == family
                                and r['condition'] == ci and r['policy'] == policy
                                and r['seed'] == (None if family == 'native' else seed)) for seed in SEEDS] for m in catalog]
                data[method] = np.asarray(values, float); assert np.isfinite(data[method]).all()
            comparisons = {}
            for family in FAMILIES:
                comparisons[f'{family}:history_vs_recent'] = paired(data[family + '/recent'], data[family + '/history'], places)
            for policy in POLICIES:
                for control in ['new_content', 'new_center', 'content_static', 'content_dynamic', 'center_dynamic', 'soft_static', 'soft_dynamic']:
                    comparisons[f'{policy}:new_soft_vs_{control}'] = paired(data[f'{control}/{policy}'], data[f'new_soft/{policy}'], places)
                for candidate, control in [('new_content', 'content_static'), ('new_content', 'content_dynamic'), ('new_center', 'center_dynamic')]:
                    comparisons[f'{policy}:{candidate}_vs_{control}'] = paired(data[f'{control}/{policy}'], data[f'{candidate}/{policy}'], places)
                for family in ('new_content', 'new_soft', 'new_center'):
                    for control in ('native', 'local'):
                        comparisons[f'{policy}:{family}_vs_{control}'] = paired(data[control], data[f'{family}/{policy}'], places)
            matrices[condition][metric] = {k: v.tolist() for k, v in data.items()}
            conditions[condition][metric] = {'means': {k: float(v.mean()) for k, v in data.items()}, 'comparisons': comparisons}
    for metric in METRICS:
        result = {}
        def history_gain(condition, family):
            d = matrices[condition][metric]
            return np.asarray(d[family + '/recent']) - np.asarray(d[family + '/history'])
        for family in FAMILIES:
            for condition in ('full', 'query_only'):
                result[f'{family}:persistent_minus_{condition}_history_gain'] = delta_summary(
                    history_gain('persistent', family) - history_gain(condition, family), places)
        for family in ('new_content', 'new_center', 'soft_dynamic', 'content_static'):
            result[f'persistent:new_soft_minus_{family}_history_gain'] = delta_summary(
                history_gain('persistent', 'new_soft') - history_gain('persistent', family), places)
        interactions[metric] = result
    training = {}
    for kind in KINDS:
        for seed in SEEDS:
            folder = OUT / f'{kind}_{seed}'
            status = json.loads((folder / 'status.json').read_text())
            rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
            training[f'{kind}_{seed}'] = {'status': status,
                'loss_by_100_steps': [float(np.mean([r['loss'] for r in rows[i:i+100]])) for i in range(0, 2000, 100)],
                'max_gradient_norm': max(r['gradient_norm'] for r in rows)}
    worker_seconds = sum(r['status']['seconds_this_invocation'] for r in training.values())
    peak = max(r['status']['peak_GiB'] for r in training.values())
    parent_analysis = json.loads((PARENT / 'analysis.json').read_text())
    result = {'experiment': plan['experiment'], 'places': places, 'conditions': conditions,
        'matrices': matrices, 'history_gain_interactions': interactions, 'training': training,
        'cost': {'training_worker_seconds': worker_seconds, 'training_worker_hours': worker_seconds/3600,
                 'peak_training_GiB': peak, 'new_predictions': new_count, 'parent_predictions': old_count,
                 'training_steps': 18000, 'pilot_steps_excluded': 6, 'new_download_bytes': 0},
        'parent_information_diagnostic': parent_analysis['information'],
        'plan_sha256': sha256(OUT / 'plan.json'), 'all_data_previously_used': True, 'new_confirmation': False}
    assert new_count == 648 and old_count == 1224
    write_json(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'cost': result['cost'],
        'primary': {k: result['conditions']['persistent']['hidden_log_mse']['comparisons'][k]
                    for k in ['new_soft:history_vs_recent', 'history:new_soft_vs_new_content',
                              'history:new_soft_vs_content_static', 'history:new_soft_vs_new_center']}}, indent=2))


if __name__ == '__main__': main()
