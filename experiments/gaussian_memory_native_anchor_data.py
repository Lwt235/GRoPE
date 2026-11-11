"""Fixed, finite native_anchor_probe1 identities; no new model or data fitting."""
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.gaussian_memory_history_data import (
    ROOT, DATA, OUT as PARENT, BASE_CACHE, FAMILIES, POLICIES, SEEDS, QUERIES,
    sha256, write_json, save_tensor)

OUT = ROOT / 'results/gaussian_memory/native_anchor_probe1'
CACHE = DATA / 'native_anchor_probe_v1'
COUNTS = {'read': 8640, 'local': 432, 'intervention': 288,
          'retention': 180, 'retention_baseline': 180}
SOURCES = ('experiments/gaussian_memory_native_anchor_data.py',
           'experiments/gaussian_memory_native_anchor_evaluate.py',
           'experiments/gaussian_memory_incremental_evaluate.py',
           'experiments/gaussian_memory_view_data.py')


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def expected_records(parent):
    for family in FAMILIES:
        for seed in SEEDS:
            for policy in POLICIES:
                case = f'{family}_{seed}_{policy}'
                for m in parent['catalog']:
                    yield 'read', case, m['id'], f'read/{case}/{m["id"]}.json'
    for seed in SEEDS:
        for m in parent['catalog']:
            case = f'local_{seed}'
            yield 'local', case, m['id'], f'reused/{case}/{m["id"]}.json'
    for intervention in ('content_permuted', 'uniform_prior'):
        case = f'soft_dynamic_{SEEDS[0]}_{intervention}'
        for m in parent['catalog']:
            yield 'intervention', case, m['id'], f'intervention/{case}/{m["id"]}.json'
    for phase in ('retention', 'retention_baseline'):
        for family in FAMILIES:
            for seed in SEEDS:
                case = f'{family}_{seed}'
                for m in parent['early_catalog']:
                    yield phase, case, m['id'], f'{phase}/{case}/{m["id"]}.json'


def prepare():
    if (OUT / 'plan.json').exists():
        plan = load_plan()
        for name, digest in plan['source_sha256'].items():
            assert sha256(ROOT / name) == digest, name
        assert sha256(ROOT / 'docs/gaussian_memory_native_anchor_probe_plan.md') == plan['document_sha256']
        return plan
    parent = json.loads((PARENT / 'plan.json').read_text())
    assert json.loads((PARENT / 'phase_completion.json').read_text())['experiment_complete']
    records = []; counts = {k: 0 for k in COUNTS}
    for phase, case, rid, relative in expected_records(parent):
        path = PARENT / 'evaluation' / relative
        r = json.loads(path.read_text()); assert r['complete']
        indices = r.get('parent_prediction_indices', list(range(len(r['rows']))))
        assert len(indices) == len(r['rows'])
        records.append({'phase': phase, 'case': case, 'id': rid, 'path': str(path),
                        'sha256': sha256(path), 'prediction_path': r['prediction_path'],
                        'prediction_sha256': r['prediction_sha256'], 'indices': indices,
                        'sample_keys': [[x['query'], x['condition']] for x in r['rows']],
                        'checkpoint_sha256': r['checkpoint_sha256']})
        counts[phase] += len(r['rows'])
    assert counts == COUNTS and sum(counts.values()) == 9720
    native_records = []
    for m in parent['catalog']:
        path = PARENT / 'evaluation/reused/native' / f'{m["id"]}.json'
        native_records.append({'id': m['id'], 'path': str(path), 'sha256': sha256(path)})
    plan = {'experiment': 'native_anchor_probe1', 'schema': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'document_sha256': sha256(ROOT / 'docs/gaussian_memory_native_anchor_probe_plan.md'),
        'parent_plan_sha256': sha256(PARENT / 'plan.json'),
        'parent_completion_sha256': sha256(PARENT / 'phase_completion.json'),
        'parent_decision_sha256': sha256(PARENT / 'method_decision.json'),
        'catalog': parent['catalog'], 'early_catalog': parent['early_catalog'],
        'pairs': parent['pairs'], 'seeds': list(SEEDS), 'families': list(FAMILIES),
        'policies': list(POLICIES), 'records': records, 'native_records': native_records,
        'expected_transforms': COUNTS, 'expected_reused_native_rows': 144,
        'formula': 'r=log(D_M)-log(D_0); delta=mean_U(r); D_out=D_0*exp(r-delta)',
        'U': 'All RGB pixels outside the fixed synthetic mask; full input uses all pixels. No sensor-valid mask or target depth.',
        'coefficient': 1, 'positive_inputs_direct_log': True, 'clipping': None,
        'primary': 'Place-equal mixed native log-MSE: before/after, anchored Soft vs anchored strong C and matched controls, anchored history vs recent.',
        'bootstrap': {'draws': 5000, 'seed': 20261110, 'units': 'paired place x model seed'},
        'retention_and_intervention_included_prospectively': True,
        'old_data_only': True, 'new_training_steps': 0, 'new_download_bytes': 0,
        'new_backbone_inferences': 0, 'new_confirmation': False,
        'source_sha256': {n: sha256(ROOT / n) for n in SOURCES}}
    write_json(OUT / 'plan.json', plan)
    for name in SOURCES:
        dest = OUT / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    return plan


if __name__ == '__main__':
    p = prepare()
    print(json.dumps({'prepared': p['experiment'], 'transforms': p['expected_transforms'],
                      'records': len(p['records']), 'native_rows': 144}), flush=True)
