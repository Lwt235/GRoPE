"""Snapshot analysis of completed paired seeds, without changing the formal run."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.gaussian_memory_coarse_analysis import numpy_scores
from experiments.gaussian_memory_coarse_data import (
    OUT, KINDS, CAPACITIES, SEEDS, plan, verify, read, write_json,
    sha256, input_path, example_keys,
)


def main():
    p = plan()
    verify(p)
    # Freeze eligibility before reading efficacy scores. Incomplete arms/seeds
    # are excluded rather than comparing their intermediate checkpoints.
    available = {
        (kind, seed): read(OUT / 'evaluation' / f'{kind}_{seed}.json')
        for kind in KINDS for seed in SEEDS
        if (OUT / 'evaluation' / f'{kind}_{seed}.json').is_file()
    }
    seeds = [s for s in SEEDS if all((k, s) in available and
             available[k, s].get('complete') for k in KINDS)]
    assert seeds, 'No complete paired seed yet'
    targets, rows, identities = {}, [], {}
    maxdiff = 0.0
    for seed in seeds:
        identities[str(seed)] = {}
        for kind in KINDS:
            ev = available[kind, seed]
            folder = OUT / f'{kind}_{seed}'
            status = read(folder / 'status.json')
            assert status['complete'] and status['steps'] == 2000
            assert sha256(folder / 'step_2000.pt') == ev['checkpoint_sha256'] == status['checkpoint_sha256']
            assert sha256(ev['prediction_path']) == ev['prediction_sha256']
            log = [json.loads(x) for x in (folder / 'training.jsonl').read_text().splitlines()]
            assert len(log) == 2000 and all(np.isfinite(r['loss']) and
                   np.isfinite(r['gradient_norm']) for r in log)
            counts = {n: sum(int(key[3]) == n for r in log for key in r['samples']) for n in CAPACITIES}
            assert counts == {256: 2000, 1024: 2000}
            identities[str(seed)][kind] = dict(
                initial=status['initial_state_sha256'], schedule=status['schedule_sha256'],
                actual_samples_sha256=hashlib.sha256(json.dumps([r['samples'] for r in log]).encode()).hexdigest(),
                checkpoint_sha256=ev['checkpoint_sha256'], prediction_sha256=ev['prediction_sha256'])
            data = torch.load(ev['prediction_path'], map_location='cpu', weights_only=False)
            assert data['keys'] == [list(k) for k in example_keys(p, 'development')]
            assert len(data['keys']) == len(data['depth_metric']) == len(ev['rows']) == 48
            for key, pred, recorded in zip(data['keys'], data['depth_metric'], ev['rows']):
                rid, q, c, n = key
                assert q == 11 and (recorded['id'], recorded['condition'], recorded['capacity']) == (rid, c, n)
                if (rid, c) not in targets:
                    inp = torch.load(input_path(rid, q, c), map_location='cpu', mmap=True, weights_only=False)
                    targets[rid, c] = {k: inp[k].clone() for k in ('target', 'valid', 'hidden')}
                    del inp
                scores = numpy_scores(pred.numpy(), targets[rid, c])
                maxdiff = max(maxdiff, max(abs(scores[m] - recorded[m]) for m in scores))
                rows.append(dict(id=rid, place=recorded['place'], condition=c, capacity=n,
                                 kind=kind, seed=seed, **scores))
        for field in ('initial', 'schedule', 'actual_samples_sha256'):
            assert len({identities[str(seed)][k][field] for k in KINDS}) == 1
    assert maxdiff < 1e-10
    lookup = {(r['seed'], r['capacity'], r['condition'], r['kind'], r['id']): r for r in rows}
    metrics = ('hidden_centered_log_mse', 'hidden_log_mse', 'hidden_bias_squared',
               'centered_log_mse', 'log_mse', 'bias_squared')
    tables, comparisons = {}, {}
    ids = p['evaluation_ids']
    def matrix(n, c, kind, metric):
        return np.array([[lookup[s, n, c, kind, rid][metric] for rid in ids] for s in seeds])
    for n in CAPACITIES:
        tables[str(n)], comparisons[str(n)] = {}, {}
        for c in p['conditions']:
            tables[str(n)][c] = {k: {m: float(matrix(n,c,k,m).mean()) for m in metrics} for k in KINDS}
            comparisons[str(n)][c] = {}
            for a, b in (('gaussian', 'center'), ('gaussian', 'sphere'), ('sphere', 'center')):
                effects = {}
                for m in metrics:
                    aa, bb = matrix(n,c,a,m), matrix(n,c,b,m)
                    effects[m] = dict(relative_error_reduction_percent=float((bb.mean()-aa.mean())/bb.mean()*100),
                        absolute_error_reduction=float(bb.mean()-aa.mean()),
                        per_seed_relative_error_reduction_percent=((bb.mean(1)-aa.mean(1))/bb.mean(1)*100).tolist(),
                        improved_places=int((aa.mean(0)<bb.mean(0)).sum()), places=len(ids),
                        per_place_relative_error_reduction_percent=((bb.mean(0)-aa.mean(0))/bb.mean(0)*100).tolist())
                comparisons[str(n)][c][f'{a}_vs_{b}'] = effects
    result = dict(preliminary=True, formal_experiment_complete=False,
        created_utc=datetime.now(timezone.utc).isoformat(), paired_seeds=seeds,
        available_completed_fits=sorted(f'{k}_{s}' for k,s in available),
        scored_predictions=len(rows), places=len(ids), tables=tables, comparisons=comparisons,
        independent_metric_max_abs_difference=maxdiff, identities=identities,
        plan_sha256=sha256(OUT/'plan.json'), script_sha256=sha256(__file__),
        all_data_previously_used=True, new_confirmation=False,
        scope='Completed matched final checkpoints only; no claim of seed stability or equivalence. No threshold, training, formula or final analysis change.',
        next_decision='Finish all three paired seeds and planned reference/cost evaluation before adopting or closing the method.')
    write_json(OUT / 'preliminary_analysis.json', result)
    write_json(OUT / 'preliminary_independent_rows.json', rows)
    for n in CAPACITIES:
        for c in p['conditions']:
            print(n,c,json.dumps(tables[str(n)][c]))
            for pair, effects in comparisons[str(n)][c].items():
                print(pair,{m:dict(gain_percent=effects[m]['relative_error_reduction_percent'],
                    improved_places=effects[m]['improved_places']) for m in metrics})
    print('PAIRED_SEEDS',seeds,'OUTPUTS',len(rows),'MAX_METRIC_DIFF',maxdiff)


if __name__ == '__main__':
    main()
