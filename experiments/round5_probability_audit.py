"""Independent NumPy arithmetic audit of sampled round-5 probability archives.

This module does not import the model, trainer, or their metric functions.
It reads archives only when explicitly invoked; importing it opens no data.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np


METHODS = ('point', 'expected', 'normalized', 'isotropic', 'fixed', 'resolvent')
SEEDS = (11, 22, 33, 44, 55)
CONTINUOUS = ('ce_sum', 'epe_argmax_sum', 'epe_soft_sum')
COUNTS = ('n_valid', 'pck8_count', 'pck4_count')
LABEL_FIELDS = ('gt_valid', 'pixcontinuous', 'key_indices', 'key_weights')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def grid_centers():
    # Independent linear-index construction, x changing fastest.
    index = np.arange(18*24)
    return np.column_stack([(index % 24)*8+3.5, (index // 24)*8+3.5])


def probability_metrics(probability, gt_valid, pixcontinuous, key_indices, key_weights):
    """Recompute additive conditional metrics with no token/key masking."""
    probability = np.asarray(probability, dtype=np.float64)
    valid = np.asarray(gt_valid)
    pixel = np.asarray(pixcontinuous, dtype=np.float64)
    indices = np.asarray(key_indices)
    weights = np.asarray(key_weights, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 432:
        raise ValueError('probability must have 432 row-major key columns')
    n = len(probability)
    if valid.shape != (n,) or valid.dtype.kind not in 'biu' or not np.isin(valid, [0, 1]).all():
        raise ValueError('gt_valid must be a Boolean/integer 0/1 query vector')
    valid = valid.astype(bool)
    if pixel.shape != (n, 2) or not np.isfinite(pixel[valid]).all():
        raise ValueError('valid target pixels must be finite [N,2]')
    if indices.shape != (n, 4) or indices.dtype.kind not in 'iu':
        raise ValueError('key_indices must be integer [N,4]')
    if weights.shape != (n, 4) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('key_weights must be finite nonnegative [N,4]')
    if ((indices < -1) | (indices >= 432)).any():
        raise ValueError('label index is outside the dense key grid')
    if (indices[valid] < 0).any() or (indices[~valid] != -1).any() or (weights[~valid] != 0).any():
        raise ValueError('valid labels and unknown-row sentinels disagree')
    if not np.allclose(weights[valid].sum(-1), 1., rtol=0, atol=2e-6):
        raise ValueError('valid target weights must sum to one')
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1+2e-6).any():
        raise ValueError('probabilities must be finite and between zero and one')
    row_sums = probability.sum(-1)
    if not np.allclose(row_sums, 1., rtol=0, atol=2e-6):
        raise ValueError('probabilities must sum to one along key columns')

    selected_indices, selected_weights = indices[valid], weights[valid]
    target_probability = probability[valid][np.arange(valid.sum())[:, None], selected_indices]
    positive = selected_weights > 0
    if (target_probability[positive] <= 0).any():
        raise ValueError('positive-weight target probability is zero; CE cannot be reconstructed from this archive')
    ce = float(-(selected_weights[positive]*np.log(target_probability[positive])).sum())
    grid = grid_centers()
    argmax = grid[probability.argmax(axis=-1)]
    soft = probability @ grid
    hard_error = np.linalg.norm(argmax[valid]-pixel[valid], axis=-1)
    soft_error = np.linalg.norm(soft[valid]-pixel[valid], axis=-1)
    return dict(ce_sum=ce, pck8_count=int((hard_error <= 8).sum()),
        pck4_count=int((hard_error <= 4).sum()), epe_argmax_sum=float(hard_error.sum()),
        epe_soft_sum=float(soft_error.sum()), n_valid=int(valid.sum()),
        maximum_probability_row_sum_error=float(np.abs(row_sums-1).max(initial=0)))


def compare_metrics(actual, reported, tolerance=1e-5):
    """Counts are exact; floating additive errors are compared per valid row."""
    for key in COUNTS:
        value = reported[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{key} must be an integer count')
        if actual[key] != value:
            raise ValueError(f'{key} differs: recomputed={actual[key]}, reported={value}')
    denominator = max(actual['n_valid'], 1)
    errors = {}
    for key in CONTINUOUS:
        value = float(reported[key])
        error = abs(actual[key]-value)/denominator
        if not np.isfinite(value) or error > tolerance:
            raise ValueError(f'{key} differs per valid row by {error:.8g}, tolerance={tolerance}')
        errors[key] = error
    return errors


def _label_digest(arrays):
    digest = hashlib.sha256()
    for field in LABEL_FIELDS:
        value = np.ascontiguousarray(arrays[field])
        digest.update(field.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def audit_runs(root, methods=METHODS, seeds=SEEDS, split='test', expected_scenes=5,
               per_scene=3, tolerance=1e-5, expected_query_rows=432):
    """Read only the explicitly selected report/archive phase, never raw data."""
    root = Path(root)
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError('duplicate method or seed request')
    if split not in ('test', 'test_zero_values'):
        raise ValueError('audit phase must be test or test_zero_values')
    records, reports, runs, labels_by_pair = [], {}, [], {}
    reference_identities = reference_sources = reference_sourcehash = None
    for method in methods:
        for seed in seeds:
            run_name = f'{method}_{seed}'
            run = root/run_name
            report_path = run/f'{split}.json'
            report_bytes = report_path.read_bytes()
            report_hash = sha_bytes(report_bytes)
            report = json.loads(report_bytes)
            rows = report['rows']
            if not rows:
                raise ValueError('empty evaluation report')
            identities = [(r['scene'], r['pair_id'], r['q_index'], r['k_index']) for r in rows]
            if len(identities) != len(set(identities)):
                raise ValueError('duplicate full-report pair identity')
            if reference_identities is None:
                reference_identities = identities
                reference_sources = report['input_sources']
                reference_sourcehash = report['sourcehash']
            elif (identities != reference_identities or report['input_sources'] != reference_sources
                  or report['sourcehash'] != reference_sourcehash):
                raise ValueError('evaluation identities/data/source version differ between runs')
            counts = Counter()
            sampled = []
            for index, row in enumerate(rows):
                if row['method'] != method or row['seed'] != seed or row['split'] != 'test':
                    raise ValueError('evaluation row method/seed/split identity mismatch')
                if counts[row['scene']] < per_scene:
                    sampled.append(index)
                counts[row['scene']] += 1
            if len(counts) != expected_scenes or any(n < per_scene for n in counts.values()):
                raise ValueError('unexpected scene count or insufficient pairs for the declared sample')
            directory = run/'predictions'/split
            paths = {p.name: p for p in directory.glob('*.npz')}
            expected_names = {f'pair_{index:04d}.npz' for index in sampled}
            if set(paths) != expected_names:
                raise ValueError('missing, extra, or misindexed probability archives')
            for index in sampled:
                path = paths[f'pair_{index:04d}.npz']
                if re.fullmatch(r'pair_\d{4,}\.npz', path.name) is None:
                    raise ValueError('invalid archive name')
                raw = path.read_bytes()
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    required = {'probability', *LABEL_FIELDS}
                    if set(archive.files) != required:
                        raise ValueError('unexpected probability archive schema')
                    arrays = {key: archive[key] for key in required}
                if arrays['probability'].shape != (expected_query_rows, 432):
                    raise ValueError('archive must retain the complete declared query/key grid')
                actual = probability_metrics(**arrays)
                errors = compare_metrics(actual, rows[index], tolerance)
                identity = identities[index]
                label_hash = _label_digest(arrays)
                if identity in labels_by_pair and labels_by_pair[identity] != label_hash:
                    raise ValueError('supervision differs between sampled method/seed archives')
                labels_by_pair[identity] = label_hash
                records.append(dict(run=run_name, report_row_index=index,
                    scene=identity[0], pair_id=identity[1], archive=str(path),
                    archive_sha256=sha_bytes(raw), matched_test_json_sha256=report_hash,
                    label_array_sha256=label_hash, recomputed=actual,
                    absolute_error_per_valid_row=errors))
            reports[str(report_path)] = report_hash
            runs.append(dict(run=run_name, full_report_pairs=len(rows),
                sampled_archives=len(sampled), scene_pair_counts=dict(counts),
                test_json_sha256=report_hash, selected_step=report['selected_step']))
    # No files may change while their audit is being accumulated.
    for path, digest in reports.items():
        if sha_bytes(Path(path).read_bytes()) != digest:
            raise ValueError('test report changed during audit')
    for record in records:
        if sha_bytes(Path(record['archive']).read_bytes()) != record['archive_sha256']:
            raise ValueError('probability archive changed during audit')
    return dict(scope='Independent NumPy arithmetic check of deterministic first-three-pairs-per-scene samples only; not all test predictions.',
        methods=list(methods), seeds=list(seeds), phase=split,
        expected_run_count=len(methods)*len(seeds), actual_run_count=len(runs),
        archives=len(records), unique_sampled_pairs=len(labels_by_pair),
        full_report_pairs_per_run=len(reference_identities or []),
        query_rows_per_archive=expected_query_rows, dense_key_columns=432,
        float_tolerance_per_valid_row=tolerance, counts_compared_exactly=True,
        maximum_absolute_error_per_valid_row={key:max(
            (r['absolute_error_per_valid_row'][key] for r in records), default=0.) for key in CONTINUOUS},
        input_sources=reference_sources, inference_sourcehash=reference_sourcehash,
        runs=runs, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('results/round5_confirmation'))
    parser.add_argument('--out', type=Path, default=Path('results/round5_probability_audit'))
    parser.add_argument('--phase', choices=['test', 'test_zero_values'], default='test')
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--expected-scenes', type=int, default=5)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('refusing overwrite of an existing probability audit')
    result = audit_runs(args.runs, args.methods, args.seeds, args.phase, args.expected_scenes)
    result['audit_source_sha256'] = sha_bytes(Path(__file__).read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(result, indent=2))
    (args.out/'README.md').write_text('# Independent sampled-probability arithmetic audit\n\n'
        f"Verified {result['actual_run_count']} complete reports and {result['archives']} sampled probability archives; "
        f"{result['full_report_pairs_per_run']} full-report pairs per run, {result['unique_sampled_pairs']} unique sampled pairs.\n\n"
        'Counts match exactly. Floating additive sums are compared after dividing by the valid-row count, '
        f"with tolerance {result['float_tolerance_per_valid_row']}. Maximum errors: {result['maximum_absolute_error_per_valid_row']}.\n\n"
        'This verifies arithmetic on the deterministic first three pairs of each scene, not every test prediction, '
        'sensor-label correctness, model selection, or the full confidence-interval analysis. No trainer/model metric function is imported. '
        'Every sampled archive and matched test report is SHA256 recorded.\n')
    print(json.dumps({key:result[key] for key in ('actual_run_count', 'archives',
        'maximum_absolute_error_per_valid_row')}))


if __name__ == '__main__':
    main()
