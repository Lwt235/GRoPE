"""Bounded-memory original inference using only six previously used fit objects."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments import native_confirmation_workflow as workflow
from experiments import round65_run as runner
from experiments.native_confirmation_gpu_resources import REV, apply_allocator_cap, ALLOCATOR_CAP_BYTES
from experiments.native_confirmation_runtime import load_cached_population, population_groups, export_one, make_model
from experiments.native_memmap_runtime import install
from experiments.native_stream_cache import atomic_json
from experiments.round65_common import ROOT, VARIANTS, sha


@torch.inference_mode()
def main():
    assert not (REV / 'validation.json').exists()
    lock, _ = workflow.checked_seal()
    prior = ROOT / 'results/native_confirmation_runtime_validation/attempt1'
    plan = json.loads((prior / 'plan.json').read_text())
    bundle = load_cached_population(plan['cache'], plan['old_rows'], plan['cache_manifest_sha256'])
    assert len(bundle['ids']) == 6
    groups = population_groups([dict(row, source='MACGS' if row['taxonomy_id'] < 30 else 'ModelNet')
                                for row in plan['old_rows']])
    runner.configure()
    install(runner)
    apply_allocator_cap()
    started = time.time()
    dense, replays = [], []
    for checkpoint in lock['models']:
        if checkpoint['seed'] != 11:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
        model = make_model(checkpoint['role']).cuda().eval()
        model.load_state_dict(payload['state_dict'])
        logits, _ = runner.evaluate(model, bundle, 'dense1024')
        torch.cuda.synchronize()
        assert np.isfinite(logits).all()
        assert all(torch.equal(value.cpu(), payload['state_dict'][name]) for name, value in model.state_dict().items())
        peak = torch.cuda.max_memory_allocated()
        assert peak < ALLOCATOR_CAP_BYTES
        dense.append(dict(role=checkpoint['role'], seed=11, objects=6, batch_size=4, tokens=1024,
                          peak_allocated_bytes=peak, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          all_logits_finite=True, model_state_unchanged=True))
        print('DENSE_MEMORY_CHECK', checkpoint['role'], peak, flush=True)
        del model, payload, logits
        torch.cuda.empty_cache()
        if checkpoint['role'] not in ('relative5', 'distance_mlp'):
            continue
        for variant in VARIANTS:
            destination = REV / 'old_fit_exports' / checkpoint['role'] / (variant + '.npz')
            export_one(bundle, checkpoint, variant, groups, destination)
            expected = prior / 'exports' / checkpoint['role'] / (variant + '.npz')
            with np.load(destination) as actual, np.load(expected) as reference:
                assert np.array_equal(actual['ids'], reference['ids'])
                assert np.array_equal(actual['labels'], reference['labels'])
                difference = float(np.max(np.abs(actual['logits'] - reference['logits'])))
                assert difference == 0
            replays.append(dict(role=checkpoint['role'], variant=variant, maximum_logit_difference=difference,
                                saved_reference_sha256=sha(expected), current_sha256=sha(destination)))
    assert len(dense) == 7 and len(replays) == 12
    record = dict(passed=True, observed_unix=time.time(), seconds=time.time()-started,
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_version=torch.__version__, allocator_cap_bytes=ALLOCATOR_CAP_BYTES,
        dense_checks=dense, old_reference_replays=replays,
        old_fit_objects=6, old_cache_manifest_sha256=plan['cache_manifest_sha256'],
        evaluation_lock_sha256=sha(workflow.OUT / 'evaluation_lock.json'),
        official_test_inputs_read=False, official_test_predictions_generated=False,
        optimizer_updates=0, resource_validation_only=True, goal_complete=False)
    atomic_json(REV / 'validation.json', record)
    print('GPU_RESOURCE_VALIDATION_PASSED', max(row['peak_allocated_bytes'] for row in dense), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(REV / ('validation_failure_' + str(time.time_ns()) + '.json'),
                    dict(type=type(error).__name__, message=str(error)))
        raise
