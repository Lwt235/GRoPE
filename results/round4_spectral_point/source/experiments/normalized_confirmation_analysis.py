"""Prospective camera replication of the frozen compact Gaussian PE candidate."""
import json
from pathlib import Path
import numpy as np
import torch

from experiments.matching_probe import digest
from experiments.query_support import use_fitted_query
from experiments.isometric_confirmation_analysis import (
    SEEDS, archived_matcher, verify_source_snapshot, aligned_real_inputs,
    real_probability_metrics, baseline_probability, real_row_identity,
    paired_camera_seed_statistics, model_probability, compare_summary,
    sha256, write_json)


ROOT = Path('results/round3_normalized_confirmation')
SCENES = ['teatime', 'waldo_kitchen', 'figurines', 'temple']
METRICS = ['cross_entropy', 'kl', 'js', 'top1_target_mass',
           'dominant_region_accuracy', 'top1_max_bin_accuracy']


def completed_training(kind, protocol):
    root = ROOT/f'training_{kind}'
    manifest = json.loads((root/'manifest.json').read_text())
    if 'elapsed_seconds' not in manifest:
        raise ValueError(f'{kind}: training still incomplete')
    config = manifest['config']
    expected = {(s, m) for s in protocol['seeds'] for m in protocol['methods']}
    actual = [(r['seed'], r['method']) for r in manifest['records']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('missing/duplicate checkpoint record')
    if ([int(s) for s in config['seeds'].split(',')] != SEEDS or
        config['methods'].split(',') != protocol['methods'] or
        config['query_geometry'] != 'fitted' or config['phase'] != 'development' or
        config['steps'] != 500 or config['resolution'] != .25 or
        config['val_every'] != 20 or config['batch'] != 16):
        raise ValueError('training differs from frozen protocol')
    # All imported numerical/data helpers must agree with archived training code.
    checks = verify_source_snapshot(root, manifest, [
        'models/geometry.py', 'experiments/query_support.py',
        ('experiments/region_pe_probe.py', ['real_dataset', 'evaluate'])])
    for name, expected_hash in protocol['source_hashes'].items():
        if manifest['source_hashes'].get(name) != expected_hash:
            raise ValueError(f'training source differs from pre-training freeze: {name}')
    cls, implementation = archived_matcher(root, manifest)
    history = json.loads((root/'training_history.json').read_text())
    selection_errors = []
    for record in manifest['records']:
        rows = [r for r in history if (r['seed'], r['method']) ==
                (record['seed'], record['method'])]
        if [r['step'] for r in rows] != list(range(0, 501, 20)):
            raise ValueError('incomplete validation history')
        selected = min(rows, key=lambda r: r['validation']['cross_entropy'])
        if selected['step'] != record['step']:
            raise ValueError('checkpoint did not use earliest minimum validation CE')
        selection_errors.append(compare_summary(selected['validation'], record['best_validation']))
    return root, manifest, cls, dict(source=checks, implementation=implementation,
        manifest_sha256=sha256(root/'manifest.json'),
        validation_selection_records_checked=len(selection_errors))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol_path = ROOT/'frozen_protocol.json'
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['input_manifest_sha256'].items():
        if sha256(name) != expected:
            raise ValueError(f'frozen input changed: {name}')
    # Require BOTH complete branches before opening fresh matcher inputs.
    training = {kind: completed_training(kind, protocol) for kind in ['raw', 'frozen']}
    rows, identities, records, checksums = {}, {}, [], {}
    verification = dict(protocol_sha256=protocol_hash, branches={},
                        source_snapshot_checks=True, checkpoint_first_replay_hashes={})
    archive = ROOT/'predictions'
    archive.mkdir(exist_ok=True)
    for kind, (trainroot, manifest, cls, verified) in training.items():
        verification['branches'][kind] = verified
        config = manifest['config']
        freshroot = Path('results')/('round3_fresh_real_data' if kind == 'raw'
                                     else 'round3_fresh_frozen_visual_data')
        data_cache = {}
        for scene in SCENES:
            for alternate in [False, True]:
                data, meta, pairs, means = aligned_real_inputs(freshroot, scene, 'test', alternate)
                data, meta = use_fitted_query(data, meta, freshroot/scene, alternate)
                key = (kind, scene, alternate)
                identities[key] = real_row_identity(pairs, data['label_valid'].numpy(), scene)
                if not identities[key]:
                    raise ValueError(f'no valid rows: {key}')
                data_cache[(scene, alternate)] = data
                verification['branches'][kind][f'{scene}/{alternate}'] = dict(
                    dataset_sha256=digest(data), metadata=meta,
                    query_cameras=len({r['q_id'] for r in identities[key]}),
                    valid_rows=len(identities[key]), pairs=len(pairs))
                for method, configuration in protocol['baseline']['selected'].items():
                    probability, _ = baseline_probability(data, means, method, configuration)
                    summary, metric_rows = real_probability_metrics(probability, data)
                    rows[(*key, method)] = {metric: np.repeat(
                        metric_rows[metric][data['label_valid'].numpy()][None], len(SEEDS), 0)
                        for metric in METRICS}
                    records.append(dict(kind=kind, scene=scene, alternate=alternate,
                                        method=method, seed=None, summary=summary))
                    file = archive/f'{kind}-{scene}-{int(alternate)}-{method}.npz'
                    np.savez_compressed(file, probability=probability,
                                        valid=data['label_valid'].numpy())
                    checksums[str(file)] = sha256(file)
        # Validation replay establishes the selected checkpoints' behavior before
        # their fresh predictions; there is no fitting or post-test selection.
        val, vm, _, _ = aligned_real_inputs(config['real_root'], 'teatime', 'validation')
        val, _ = use_fitted_query(val, vm, Path(config['real_root'])/'teatime')
        train, tm, _, _ = aligned_real_inputs(config['real_root'], 'teatime', 'train')
        train, _ = use_fitted_query(train, tm, Path(config['real_root'])/'teatime')
        for seed in SEEDS:
            saved = manifest['datasets'][str(seed)]
            if digest(val) != saved['validation'] or digest(train) != saved['train']:
                raise ValueError('training/validation data replay mismatch')
        val_errors = []
        for record in manifest['records']:
            seed, method = record['seed'], record['method']
            checkpoint = trainroot/'checkpoints'/f'{seed}-{method}.pt'
            before = sha256(checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['method'] != method or payload['resolution'] != .25:
                raise ValueError('checkpoint metadata mismatch')
            model = cls(method, config['features'], config['resolution'])
            model.load_state_dict(payload['state_dict'], strict=True)
            model.eval()
            val_summary, _ = real_probability_metrics(model_probability(model, val), val)
            val_errors.append(compare_summary(val_summary, record['best_validation']))
            for scene in SCENES:
                for alternate in [False, True]:
                    data = data_cache[(scene, alternate)]
                    probability = model_probability(model, data)
                    summary, metric_rows = real_probability_metrics(probability, data)
                    key = (kind, scene, alternate, method)
                    if key not in rows:
                        rows[key] = {m: np.empty((len(SEEDS), int(data['label_valid'].sum())))
                                     for m in METRICS}
                    for metric in METRICS:
                        rows[key][metric][SEEDS.index(seed)] = metric_rows[metric][data['label_valid'].numpy()]
                    records.append(dict(kind=kind, scene=scene, alternate=alternate,
                                        method=method, seed=seed, summary=summary))
                    file = archive/f'{kind}-{scene}-{int(alternate)}-{seed}-{method}.npz'
                    np.savez_compressed(file, probability=probability,
                                        valid=data['label_valid'].numpy())
                    checksums[str(file)] = sha256(file)
            if sha256(checkpoint) != before:
                raise ValueError('checkpoint changed during replay')
            verification['checkpoint_first_replay_hashes'][str(checkpoint)] = before
            print('REPLAYED', kind, seed, method, flush=True)
        verification['branches'][kind]['maximum_validation_metric_replay_error'] = max(
            v for errors in val_errors for v in errors.values())
        if sha256(trainroot/'manifest.json') != verified['manifest_sha256']:
            raise ValueError('training manifest changed during replay')
    comparisons, means = {}, {}
    controls = [m for m in protocol['methods'] if m != 'normalized_shape']
    controls += list(protocol['baseline']['selected'])
    groups = {scene: [scene] for scene in SCENES}
    groups.update(external=['figurines', 'temple'], all=SCENES)
    for kind in ['raw', 'frozen']:
        for alternate in [False, True]:
            for group, scenes in groups.items():
                prefix = f'{kind}/{int(alternate)}/{group}'
                ids = [i for scene in scenes for i in identities[(kind, scene, alternate)]]
                def values(method, metric):
                    return np.concatenate([rows[(kind, scene, alternate, method)][metric]
                                           for scene in scenes], axis=1)
                for method in protocol['methods']+list(protocol['baseline']['selected']):
                    means[f'{prefix}/{method}'] = dict(
                        row_weighted={m: float(values(method, m).mean()) for m in METRICS},
                        scene_macro={m: float(np.mean([rows[(kind, s, alternate, method)][m].mean()
                                                       for s in scenes])) for m in METRICS},
                        valid_rows=len(ids), query_cameras=len({(r['scene'], r['q_id']) for r in ids}))
                for control in controls:
                    comparisons[f'{prefix}/{control}'] = {metric: paired_camera_seed_statistics(
                        values('normalized_shape', metric), values(control, metric), ids)
                        for metric in ['cross_entropy', 'js', 'top1_target_mass']}
    if sha256(protocol_path) != protocol_hash:
        raise ValueError('frozen protocol changed during replay')
    verification['prediction_sha256'] = checksums
    verification['analysis_source_sha256'] = sha256(__file__)
    gates = {}
    for kind in ['raw', 'frozen']:
        for alternate in [False, True]:
            p = f'{kind}/{int(alternate)}/external/'
            def passes(control):
                result = comparisons[p+control]['cross_entropy']
                return bool(result['ci95_seed_camera'][1] < 0 and
                            sum(v < 0 for v in result['seed_differences'].values()) >= 4)
            gates[f'{kind}/{int(alternate)}'] = dict(
                support=passes('normalized_fixed'),
                anisotropy=passes('normalized_trace') and passes('normalized_area'),
                versus_projected_rope=passes('projected_rope'),
                versus_strong_projected_rbf=passes('projected_center_rbf'))
    write_json(ROOT/'fresh_summary.json', dict(protocol_sha256=protocol_hash,
        means=means, comparisons=comparisons, gates=gates, records=records))
    write_json(ROOT/'verification.json', verification)
    lines = ['# Frozen normalized Gaussian PE: fresh-camera replication', '',
        'Five seeds fit Teatime train only. Validation CE selects each checkpoint before fresh prediction. '
        'Both raw RGB and the predeclared frozen ResNet18 representation are reported. '
        'All methods receive the same fitted query geometry. Fixed/trace/area/shuffle change key covariance only.', '',
        'Primary group: previously unused figurines + temple camera pairs. '
        'Differences below are normalized_shape minus control; negative CE favors full support. '
        'The 95% intervals use paired seed/query-camera bootstrap, stratified by scene. '
        'The fixed geometric baseline is repeated across seeds only to form paired differences.', '']
    for kind in ['raw', 'frozen']:
        for alternate in [False, True]:
            lines += [f'## {kind}, GS parity {int(alternate)}', '',
                '| Control | External CE | Full-minus-control CE | 95% CI | Four-scene macro CE |',
                '|---|---:|---:|---|---:|']
            for method in protocol['methods']+list(protocol['baseline']['selected']):
                entry = means[f'{kind}/{int(alternate)}/external/{method}']
                macro = means[f'{kind}/{int(alternate)}/all/{method}']['scene_macro']['cross_entropy']
                if method == 'normalized_shape':
                    difference, ci = '—', '—'
                else:
                    result = comparisons[f'{kind}/{int(alternate)}/external/{method}']['cross_entropy']
                    difference = f"{result['mean']:+.6f}"
                    ci = str([round(v, 6) for v in result['ci95_seed_camera']])
                lines.append(f"| {method} | {entry['row_weighted']['cross_entropy']:.6f} | {difference} | {ci} | {macro:.6f} |")
            lines += ['', 'Frozen gates: '+json.dumps(gates[f'{kind}/{int(alternate)}']), '']
    lines += ['## Scope and verification', '',
        '203 pairs were selected before matcher evaluation by excluding all previously exported camera IDs. '
        'Tea/kitchen have few remaining query cameras; per-scene results and scene-macro means are included in JSON. '
        'The known scenes, GS reconstruction and SfM may already use these source photographs. '
        'This is conditional regional transfer against sparse observed tracks, not unseen-scene reconstruction, '
        'dense visibility or rejection truth. Five training seeds are not five independent datasets. '
        'No test-based checkpoint, feature, partition or bandwidth selection is allowed.', '',
        'Verification executes archived model code, checks every frozen input/source hash, '
        'validates all 100 checkpoints against complete validation selection histories and replays validation metrics. '
        'Every evaluated NPZ is checked against its manifest. Probabilities and their hashes are archived. '
        'See fresh_summary.json for all scene/partition/seed results, proper-score and target-mass comparisons.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(gates), flush=True)


if __name__ == '__main__':
    main()
