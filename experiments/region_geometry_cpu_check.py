"""Read-only CPU evaluation of the six already sealed regional checkpoints."""
from pathlib import Path
import time

import numpy as np
import torch

from experiments import region_geometry_controls as source
from experiments.round65_common import read, write, sha

OUT = source.OUT / 'cpu_evaluation_01'


@torch.no_grad()
def evaluate(model, cache):
    model.eval()
    size = source.engine.SETTINGS['eval_batch_size']
    logits = np.concatenate([model(*source.engine.inputs(cache, slice(start, start + size), device='cpu')).numpy()
                             for start in range(0, len(cache['ids']), size)])
    return logits, source.engine.grouped_metrics(logits, cache)


def main():
    source.bind()
    source.check()
    source.engine.configure()
    assert not OUT.exists()
    selected = read(source.OUT / 'selected.json')
    assert selected['all_six_selected_before_eval'] and len(selected['records']) == 6
    OUT.mkdir()
    write(OUT / 'plan.json', dict(created_unix=time.time(), source_sha256=sha(Path(__file__)),
        lock_sha256=sha(source.OUT / 'lock.json'), selected_sha256=sha(source.OUT / 'selected.json'),
        scope='Hardware-only read-only evaluation of all six sealed models, old validation and both old evaluation conditions; no fitting, selection, metric or data change.',
        hypothesis_unchanged='Use the original regional geometry control plan and comparisons.',
        purpose='Obtain the complete scientific comparison while GPU work waits for resources; compare CPU old-validation outputs to saved GPU outputs and independently recalculate metrics.',
        reporting='Preserve all seeds, scenes, four strata and both conditions. Record numerical and argmax differences; do not conceal them or tune a model.',
        device='cpu', threads=4, fresh_confirmation=False, goal_complete=False))
    caches = {name: source.data_module.load_cache(name) for name in ('val', 'eval_base', 'eval_sparse256')}
    records, replay = [], []
    for choice in selected['records']:
        role, seed = choice['role'], choice['seed']
        run = source.OUT / 'runs' / role / str(seed)
        assert sha(run / 'best.pt') == choice['checkpoint_sha256']
        model = source.GeometryControlMatcher(role).cpu()
        model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        started = time.time()
        logits, metrics = evaluate(model, caches['val'])
        with np.load(run / 'validation.npz') as saved:
            assert np.array_equal(saved['ids'], caches['val']['ids'])
            error = float(abs(saved['logits'] - logits).max())
            changes = int((saved['logits'].argmax(-1) != logits.argmax(-1)).sum())
        replay.append(dict(role=role, seed=seed, maximum_logit_error=error,
                           prediction_changes=changes, checkpoint_sha256=choice['checkpoint_sha256']))
        dest = OUT / role / str(seed)
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'validation.npz', logits=logits, ids=caches['val']['ids'])
        for condition in source.CONDITIONS:
            cache = caches['eval_' + condition]
            logits, metrics = evaluate(model, cache)
            independent = source.engine.metric_torch(logits, cache['overlap'], cache['world_centers'])
            delta = max(abs(independent[k] - metrics['micro'][k])
                        for k in ('region_recall', 'center_pck', 'soft_epe', 'ce'))
            assert delta < 1e-10 and metrics['micro']['valid_queries'] == 20422
            path = dest / (condition + '.npz')
            np.savez_compressed(path, logits=logits, ids=cache['ids'])
            records.append(dict(role=role, seed=seed, condition=condition, metrics=metrics,
                prediction_sha256=sha(path), checkpoint_sha256=choice['checkpoint_sha256'],
                independent_metric_error=delta))
        write(OUT / 'progress.json', dict(completed_models=len(replay), records=records,
                                         validation_replay=replay, observed_unix=time.time()))
        print('CPU_MODEL_COMPLETE', role, seed, 'seconds', time.time() - started,
              'validation_max_error', error, 'argmax_changes', changes, flush=True)
    write(OUT / 'result.json', dict(records=records, validation_replay=replay, device='cpu',
        selected_sha256=sha(source.OUT / 'selected.json'), plan_sha256=sha(OUT / 'plan.json'),
        original_gpu_evaluation_is_separate=True, fresh_confirmation=False, goal_complete=False))
    print('CPU_EVALUATION_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
