"""Reuse the native-input arithmetic check on available ModelNet fitting data."""
import argparse
import time
import numpy as np  # Preserve the compatible host import order.
from experiments import round65_native_preflight as native
from experiments.round65_common import OUT, PREP, RAW, CACHE_ROOT, CONFIG, ROLES, checked, read, write, sha
from experiments.round65_data import split_rows


DEST = OUT / 'native_modelnet_preflight1'


def plan():
    checked()
    fit = split_rows(read(PREP / 'train_catalog.json')['rows'])['fit']
    available = sorted((row for row in fit if row['taxonomy_id'] >= 30
        and (RAW / row['file_path']).is_file()
        and (RAW / row['file_path']).stat().st_size == row['size']), key=lambda row: row['file_path'])
    if len(available) < CONFIG['batch_size']:
        raise ValueError('sixteen complete ModelNet fitting files are required')
    DEST.mkdir(parents=True, exist_ok=False)
    write(DEST / 'plan.json', dict(created_unix=time.time(), source_sha256=sha(native.__file__),
        wrapper_source_sha256=sha(__file__), lock_sha256=sha(OUT / 'lock.json'),
        catalog_sha256=sha(PREP / 'train_catalog.json'), rows=available[:CONFIG['batch_size']],
        available_modelnet_fit_files=len(available),
        input_selection='lexicographically first16 currently complete original R65 fit entries with taxonomy_id>=30; no label-based exclusion',
        hypothesis='the unchanged R65 converter and five models also support finite native ModelNet input arithmetic',
        competing_explanation='ModelNet parameter scales or file layout expose an implementation limitation absent from MACGS checks',
        fixed_comparison=dict(roles=ROLES, seed=11, batch=16, tokens=512, native_cache_tokens=2048,
            same_inputs=True, same_common_initialization=True, objective='mean squared random-initialization logits',
            optimizer_steps=0, labels_used=False, reused_diagnostic_source_sha256=sha(native.__file__)),
        decisions=dict(pass_case='retain the unchanged implementation for full R65; no task-effectiveness claim',
            failure_case='preserve failure and diagnose conversion/arithmetic; do not exclude failing inputs',
            floor_diagnostics='descriptive only; no new threshold or floor tuning'),
        limitations=['sixteen availability-selected fitting objects; not source-wide or class-wide evidence',
            'no task training, accuracy measurement, validation/dev_eval access or fresh confirmation'],
        config=CONFIG, numerical_arrays_read_at_plan=False, full_controller_resumed=False, goal_complete=False))
    print('MODELNET_INPUT_PLAN_CREATED', len(available), 'available fitting files;16 selected', flush=True)


def run():
    specification = read(DEST / 'plan.json')
    if (specification['wrapper_source_sha256'] != sha(__file__)
            or specification['catalog_sha256'] != sha(PREP / 'train_catalog.json')
            or specification['config'] != CONFIG):
        raise ValueError('ModelNet diagnostic specification changed')
    native.DEST = DEST
    native.CACHE_ROOT = CACHE_ROOT / 'modelnet_input_diagnostic1'
    native.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'run'))
    (plan if parser.parse_args().action == 'plan' else run)()
