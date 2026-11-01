"""Audited real-data analysis of Round 4's retrained spectral PE controls.

All evaluation cameras were examined in Round 3: this is a descriptive extension.
Both complete training branches, complete validation histories, data digests and
checkpoint validation replay are required before any evaluation NPZ is opened.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.matching_probe import digest
from experiments.query_support import use_fitted_query
from experiments.isometric_confirmation_analysis import (
    SEEDS, aligned_real_inputs, real_probability_metrics, real_row_identity,
    paired_camera_seed_statistics, model_probability, compare_summary,
    sha256, write_json, verify_source_snapshot)


METHODS = ['gaussian_full', 'gaussian_qonly', 'gaussian_konly', 'gaussian_none',
           'gaussian_trace', 'gaussian_area', 'gaussian25_full', 'gaussian50_full',
           'gaussian200_full', 'resolvent_full', 'resolvent_none', 'resolvent_trace', 'resolvent_area']
SCALES = ['gaussian25_full', 'gaussian50_full', 'gaussian_full', 'gaussian200_full']
SCENES = ['teatime', 'waldo_kitchen', 'figurines', 'temple']
METRICS = ['cross_entropy', 'kl', 'js', 'top1_target_mass', 'dominant_region_accuracy', 'top1_max_bin_accuracy']


def wait_until_complete(roots, seconds):
    deadline = time.monotonic()+seconds
    while True:
        state = {}
        for kind, root in roots.items():
            path = root/'manifest.json'
            if not path.exists():
                state[kind] = {'records': 0, 'complete': False}
                continue
            manifest = json.loads(path.read_text())
            state[kind] = {'records': len(manifest.get('records', [])),
                           'complete': 'elapsed_seconds' in manifest}
        if all(value['complete'] for value in state.values()):
            return
        print('WAITING_FOR_TRAINING', json.dumps(state), flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError('both complete training branches are required before evaluation')
        time.sleep(min(10., max(0., deadline-time.monotonic())))


def completed_branch(root):
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if 'elapsed_seconds' not in manifest:
        raise ValueError('training is incomplete')
    config = manifest['config']
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    actual = [(record['seed'], record['method']) for record in manifest['records']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('missing or duplicate checkpoint records')
    if ([int(seed) for seed in config['seeds'].split(',')] != SEEDS
        or config['methods'].split(',') != METHODS or config['phase'] != 'development'
        or config['query_geometry'] != 'fitted' or config['domain'] != 'real'
        or config['resolution'] != .25 or config['steps'] != 500
        or config['val_every'] != 20 or config['batch'] != 16):
        raise ValueError('training configuration differs from the predeclared extension')
    freeze = json.loads((root/'frozen_protocol.json').read_text())
    if freeze['config'] != config or freeze['methods'] != METHODS:
        raise ValueError('training manifest differs from frozen extension protocol')
    for name, expected_hash in freeze['source_hashes'].items():
        if manifest['source_hashes'].get(name) != expected_hash:
            raise ValueError(f'training source differs from frozen protocol: {name}')
    verification = verify_source_snapshot(root, manifest, [
        'models/geometry.py', 'models/gaussian_pe.py', 'experiments/query_support.py',
        'experiments/round4_spectral_probe.py',
        ('experiments/region_pe_probe.py', ['real_dataset', 'evaluate'])])
    path = root/'source/models/gaussian_spectral_pe.py'
    checksum = sha256(path)
    if checksum != manifest['source_hashes']['models/gaussian_spectral_pe.py']:
        raise ValueError('archived spectral implementation changed')
    specification = importlib.util.spec_from_file_location('_round4_spectral_'+checksum[:16], path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    verification.update(archived_spectral_model=str(path), archived_spectral_sha256=checksum,
                        manifest_sha256=sha256(manifest_path),
                        frozen_protocol_sha256=sha256(root/'frozen_protocol.json'))
    history = json.loads((root/'training_history.json').read_text())
    for record in manifest['records']:
        rows = [row for row in history if (row['seed'], row['method']) == (record['seed'], record['method'])]
        if [row['step'] for row in rows] != list(range(0, 501, 20)):
            raise ValueError('incomplete validation history including step zero')
        selected = min(rows, key=lambda row: row['validation']['cross_entropy'])
        if selected['step'] != record['step']:
            raise ValueError('checkpoint is not earliest minimum validation CE')
        compare_summary(selected['validation'], record['best_validation'])
    scale_validation = {method: float(np.mean([r['best_validation']['cross_entropy']
                         for r in manifest['records'] if r['method'] == method])) for method in SCALES}
    selected_scale = min(SCALES, key=scale_validation.get)
    verification['scale_selection'] = dict(selected=selected_scale, aggregate_validation_ce=scale_validation,
        rule='minimum old Teatime validation CE averaged over all five training seeds; tie order .25,.5,1,2; evaluation not used')
    return manifest, module.GaussianSpectralMatcher, verification


def real_inputs(root, scene, split, alternate=False):
    data, meta, pairs, means = aligned_real_inputs(root, scene, split, alternate)
    data, meta = use_fitted_query(data, meta, Path(root)/scene, alternate)
    return data, meta, pairs


def compare_old_checkpoint(root, kind, record, payload):
    mapping = {'gaussian_full': 'normalized_shape', 'gaussian_qonly': 'normalized_fixed'}
    if record['method'] not in mapping:
        return None
    old_path = Path(f'results/round3_normalized_confirmation/training_{kind}/checkpoints/{record["seed"]}-{mapping[record["method"]]}.pt')
    old = torch.load(old_path, map_location='cpu', weights_only=True)
    if set(old['state_dict']) != set(payload['state_dict']):
        raise ValueError('equivalent old/new models have different state keys')
    errors = {name: float((value-old['state_dict'][name]).abs().max())
              for name, value in payload['state_dict'].items()}
    maximum = max(errors.values())
    if maximum != 0.:
        raise ValueError(f'expected old/new deterministic checkpoint equality failed: {kind} {record["seed"]} {record["method"]} {maximum}')
    return dict(old_method=mapping[record['method']], old_checkpoint=str(old_path),
                old_checkpoint_sha256=sha256(old_path), maximum_state_error=maximum)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    roots = {kind: Path(f'results/round4_spectral_real_{kind}') for kind in ['raw', 'frozen']}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/'analysis_protocol.json'
    design = dict(stage='written before waiting for both complete extension branches',
        data_status='all four existing fresh-camera datasets have already been evaluated in Round 3; descriptive extension, not untouched confirmation',
        methods=METHODS, seeds=SEEDS, scenes=SCENES, primary_group=['figurines', 'temple'],
        partitions='both GS parity partitions reported separately, never select favorable partition',
        gaussian_scale='minimum aggregate five-seed Teatime validation CE among .25,.5,1,2 physical covariance scales, before any extension test evaluation',
        controls='Gaussian factorial q/k support; symmetric none/trace/area; resolvent same covariance and normalization; all methods share labels/validity',
        statistics='paired crossed training-seed/query-camera bootstrap stratified by fixed scene; per-scene and scene macro results retained',
        no_joint_decision='point robustness results are analyzed separately by the root agent; this file cannot establish the combined replacement criterion',
        source_sha256=sha256(__file__), config=vars(args))
    if protocol_path.exists():
        raise FileExistsError('use a new analysis output directory to preserve recorded design')
    write_json(protocol_path, design)
    wait_until_complete(roots, args.wait_seconds)
    # Completeness, source and history checks for BOTH branches precede test NPZ.
    branches = {kind: completed_branch(root) for kind, root in roots.items()}
    started = time.time()
    verification = dict(branches={}, checkpoints={}, old_equivalence={}, analysis_protocol_sha256=sha256(protocol_path))
    identities, values, records, probability_hashes = {}, {}, [], {}
    archive = out/'predictions'
    archive.mkdir(exist_ok=True)
    for kind, (manifest, cls, checked) in branches.items():
        root = roots[kind]
        config = manifest['config']
        verification['branches'][kind] = checked
        train, tm, _ = real_inputs(config['real_root'], 'teatime', 'train')
        val, vm, _ = real_inputs(config['real_root'], 'teatime', 'validation')
        for seed in SEEDS:
            saved = manifest['datasets'][str(seed)]
            if digest(train) != saved['train'] or digest(val) != saved['validation']:
                raise ValueError('training/validation digest replay mismatch')
        verification['branches'][kind]['validation_dataset_sha256'] = digest(val)
        # All selected checkpoints replay validation BEFORE test inputs open.
        loaded = []
        for record in manifest['records']:
            checkpoint = root/'checkpoints'/f'{record["seed"]}-{record["method"]}.pt'
            checksum = sha256(checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if (payload['method'] != record['method'] or payload['features'] != config['features']
                or payload['resolution'] != config['resolution']):
                raise ValueError('checkpoint metadata mismatch')
            model = cls(record['method'], payload['features'], payload['resolution']).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            summary, _ = real_probability_metrics(model_probability(model, val), val)
            errors = compare_summary(summary, record['best_validation'])
            verification['checkpoints'][str(checkpoint)] = dict(sha256=checksum, validation_errors=errors)
            equivalent = compare_old_checkpoint(root, kind, record, payload)
            if equivalent is not None:
                verification['old_equivalence'][f'{kind}/{record["seed"]}/{record["method"]}'] = equivalent
            loaded.append((record, model, equivalent))
        fresh_root = Path('results')/('round3_fresh_real_data' if kind == 'raw' else 'round3_fresh_frozen_visual_data')
        datasets = {}
        for scene in SCENES:
            for alternate in [False, True]:
                key = (kind, scene, alternate)
                data, meta, pairs = real_inputs(fresh_root, scene, 'test', alternate)
                datasets[(scene, alternate)] = data
                identities[key] = real_row_identity(pairs, data['label_valid'].numpy(), scene)
                verification['branches'][kind][f'{scene}/{alternate}'] = dict(dataset_sha256=digest(data),
                    metadata=meta, valid_rows=len(identities[key]),
                    query_cameras=len({row['q_id'] for row in identities[key]}))
        for record, model, equivalent in loaded:
            seed, method = record['seed'], record['method']
            for (scene, alternate), data in datasets.items():
                probability = model_probability(model, data)
                summary, rows = real_probability_metrics(probability, data)
                key = (kind, scene, alternate, method)
                if key not in values:
                    values[key] = {metric: np.empty((len(SEEDS), int(data['label_valid'].sum()))) for metric in METRICS}
                for metric in METRICS:
                    values[key][metric][SEEDS.index(seed)] = rows[metric][data['label_valid'].numpy()]
                records.append(dict(kind=kind, scene=scene, alternate=alternate,
                                    seed=seed, method=method, summary=summary))
                target = archive/f'{kind}-{scene}-{int(alternate)}-{seed}-{method}.npz'
                np.savez_compressed(target, probability=probability, valid=data['label_valid'].numpy())
                probability_hashes[str(target)] = sha256(target)
                if equivalent is not None:
                    old_path = Path('results/round3_normalized_confirmation/predictions')/f'{kind}-{scene}-{int(alternate)}-{seed}-{equivalent["old_method"]}.npz'
                    with np.load(old_path) as old:
                        if not np.array_equal(old['valid'], data['label_valid'].numpy()):
                            raise ValueError('old equivalent model has different evaluation rows')
                        maximum = float(np.max(np.abs(probability-old['probability'])))
                    if maximum > 2e-7:
                        raise ValueError('old/new equivalent predictions diverged')
                    equivalent.setdefault('prediction_replay', {})[f'{scene}/{alternate}'] = maximum
            print('REPLAYED', kind, seed, method, flush=True)
        if sha256(root/'manifest.json') != checked['manifest_sha256']:
            raise ValueError('training manifest changed during analysis')
    comparisons, means = {}, {}
    groups = {scene: [scene] for scene in SCENES}
    groups.update(external=['figurines', 'temple'], all=SCENES)
    comparison_pairs = [('gaussian_full', 'gaussian_'+control) for control in ['qonly', 'konly', 'none', 'trace', 'area']]
    comparison_pairs += [('gaussian_konly', 'gaussian_full'), ('resolvent_full', 'gaussian_full'),
                         ('resolvent_full', 'gaussian_selected')]
    comparison_pairs += [('resolvent_full', control) for control in ['resolvent_none', 'resolvent_trace', 'resolvent_area']+SCALES]
    comparison_pairs = list(dict.fromkeys(comparison_pairs))
    for kind in ['raw', 'frozen']:
        selected = verification['branches'][kind]['scale_selection']['selected']
        for alternate in [False, True]:
            for group, scenes in groups.items():
                prefix = f'{kind}/{int(alternate)}/{group}'
                ids = [row for scene in scenes for row in identities[(kind, scene, alternate)]]
                def get(method, metric):
                    method = selected if method == 'gaussian_selected' else method
                    return np.concatenate([values[(kind, scene, alternate, method)][metric] for scene in scenes], axis=1)
                for method in METHODS+['gaussian_selected']:
                    actual = selected if method == 'gaussian_selected' else method
                    means[f'{prefix}/{method}'] = dict(actual_method=actual,
                        row_weighted={metric: float(get(method, metric).mean()) for metric in METRICS},
                        scene_macro={metric: float(np.mean([values[(kind, scene, alternate, actual)][metric].mean() for scene in scenes])) for metric in METRICS},
                        valid_rows=len(ids), query_cameras=len({(row['scene'], row['q_id']) for row in ids}))
                for candidate, control in comparison_pairs:
                    key = f'{prefix}/{candidate}_minus_{control}'
                    comparisons[key] = {metric: paired_camera_seed_statistics(get(candidate, metric), get(control, metric), ids)
                        for metric in ['cross_entropy', 'js', 'top1_target_mass']}
    for path, data in verification['checkpoints'].items():
        if sha256(path) != data['sha256']:
            raise ValueError('checkpoint changed during analysis')
    verification.update(prediction_sha256=probability_hashes, analysis_source_sha256=sha256(__file__))
    summary = dict(scope=design['data_status'], means=means, comparisons=comparisons, records=records,
                   selected_gaussian_scale={kind: verification['branches'][kind]['scale_selection'] for kind in ['raw','frozen']},
                   elapsed_seconds=time.time()-started)
    write_json(out/'summary.json', summary)
    write_json(out/'verification.json', verification)
    row_archive = {f'{kind}/{scene}/{int(alternate)}/{method}/{metric}': row
                   for (kind, scene, alternate, method), metrics in values.items() for metric, row in metrics.items()}
    np.savez_compressed(out/'row_metrics.npz', **row_archive)
    lines = ['# Round 4: retrained covariance controls and spectral family', '',
        'Descriptive extension on already examined real cameras. Five matched training seeds; both visual inputs; '
        'all four scenes and both GS partitions are retained. Primary group is figurines + temple.', '',
        'All 130 checkpoints replayed old validation before extension evaluation. Source snapshots, complete '
        'checkpoint-selection histories, training/validation data hashes and evaluation NPZ hashes are verified. '
        'Gaussian covariance scale is selected only by five-seed mean Teatime validation CE.', '',
        '| Visual input | GS partition | Method | CE | JS | Top1 target mass |',
        '|---|---|---|---:|---:|---:|']
    for kind in ['raw','frozen']:
        for alternate in [False,True]:
            for method in METHODS+['gaussian_selected']:
                row = means[f'{kind}/{int(alternate)}/external/{method}']['row_weighted']
                lines.append(f"| {kind} | {int(alternate)} | {method} | {row['cross_entropy']:.6f} | {row['js']:.6f} | {row['top1_target_mass']:.6f} |")
    lines += ['', 'Differences and 95% paired seed/query-camera intervals are in summary.json. '
        'Negative candidate-minus-control CE favors the candidate. This camera bootstrap treats scenes '
        'as fixed strata, and five model seeds are not five independent datasets. Known GS/SfM may use '
        'evaluation photos; labels are conditional sparse-track transfer, not dense occlusion or NVS truth. '
        'The joint replacement decision additionally requires the separately analyzed point/noise guardrails.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE', summary['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/round4_spectral_real_analysis')
    parser.add_argument('--wait-seconds', type=int, default=0)
    main(parser.parse_args())
