"""Bounded pre-run diagnosis of bitwise GPU update comparison failure."""
import json
import numpy as np
import torch
from experiments.gaussian_memory_history_data import OUT, load_plan, checked_load, device_tree, write_json
from experiments.gaussian_memory_history_geometry import widen_source_columns
from models.gaussian_memory_incremental import update_memory as old_update
from models.gaussian_memory_history import update_memory


def main():
    pair = load_plan()['pairs'][0]; rid = pair['early_id']
    memory = device_tree(checked_load(rid, 'initial.pt', pair)['memory'])
    obs = device_tree(checked_load(rid, 'updates.pt', pair)[4])
    results = [old_update(memory, obs, [obs]), old_update(memory, obs, [obs]),
               update_memory(memory, obs, [obs], allowed_frames=(4, 6, 8, 10)),
               update_memory(widen_source_columns(memory, 24), obs, [obs], allowed_frames=(4, 6, 8, 10))]
    comparisons = {}
    for label, (other, log) in zip(('old_repeat', 'new_guard', 'widened_sources'), results[1:]):
        diff = {}
        for k, value in results[0][0].items():
            test = other[k][:, :12] if label == 'widened_sources' and k in ('sources', 'conflicts') else other[k]
            diff[k] = {'exact': torch.equal(value, test), 'max_absolute': float((value.double() - test.double()).abs().max())}
        comparisons[label] = {'fields': diff, 'event_log_exact': log == results[0][1]}
    result = {'scope': 'Pre-run failure diagnosis; no history states or model predictions generated.', 'comparisons': comparisons}
    write_json(OUT / 'preflight_numerical_failure_diagnostic.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
