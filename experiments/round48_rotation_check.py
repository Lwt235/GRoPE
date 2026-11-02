"""Bounded post-result check of R48's single, already-used tilt condition.

No angle sweep, fitting, checkpoint selection, or fresh-data access. The plan
and this source are saved before loading task arrays or replaying any model.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch

from experiments.round48_common import ROOT, OUT, checked, read, write, sha, hash_array
from experiments.round48_data import load, evaluation_points
from experiments.round48_train import configure
from models.round48_point_transformer import make_model


DEST = ROOT/'results/round48_modelnet/rotation_check1'
ROLES = ('conditional', 'rope', 'nope')


def coordinate_reference(points):
    """Apply Rx(30 degrees), then Rz(45 degrees), without a matrix product."""
    x, y, z = np.moveaxis(np.asarray(points, dtype=np.float64), -1, 0)
    cx, sx, cz, sz = np.sqrt(3)/2, .5, np.sqrt(.5), np.sqrt(.5)
    yr, zr = cx*y-sx*z, sx*y+cx*z
    return np.stack((cz*x-sz*yr, sz*x+cz*yr, zr), axis=-1)


@torch.no_grad()
def prediction(model, points):
    outputs = []
    for start in range(0, len(points), 16):
        x = torch.from_numpy(np.ascontiguousarray(points[start:start+16], dtype=np.float32))
        outputs.append(model(x).numpy())
    return np.concatenate(outputs)


def main():
    started = time.monotonic()
    lock = checked(OUT)
    if read(OUT/'queue_status.json')['state'] != 'complete':
        raise ValueError('finish the frozen experiment first')
    if DEST.exists():
        raise ValueError('preserve previous diagnostic')
    seal = read(OUT/'audit_training/completion.json')
    verification = read(OUT/'metrics/verification.json')
    inputs = {str(OUT/name): sha(OUT/name) for name in (
        'lock.json', 'audit_training/completion.json', 'metrics/summary.json',
        'metrics/verification.json', 'data/manifest.json')}
    for role in ROLES:
        row = next(r for r in seal['records'] if r['role'] == role and r['seed'] == 11)
        checkpoint = Path(row['path'])/'best.pt'
        predictions = Path(row['path'])/'dev_eval.npz'
        if sha(checkpoint) != row['checkpoint_sha256']:
            raise ValueError('sealed checkpoint changed')
        if sha(predictions) != verification['input_hashes'][str(predictions)]:
            raise ValueError('audited predictions changed')
        inputs[str(checkpoint)] = row['checkpoint_sha256']
        inputs[str(predictions)] = verification['input_hashes'][str(predictions)]
    DEST.mkdir()
    (DEST/'source.py').write_bytes(Path(__file__).read_bytes())
    plan = dict(
        question='Does the common SO3 collapse arise from a wrong/nonrigid transform or replay mismatch?',
        competing_explanation='The declared rigid tilt is correct and these fixed models are orientation-sensitive.',
        trigger='Completed R48 reports 4.50--8.00 percent OA under its single prescribed SO3 tilt.',
        comparison='All 965 used development input clouds: frozen transform versus coordinate-wise double precision reference; one fixed seed and three roles replayed on 32 identity-selected objects.',
        outcomes='A material mismatch requires preserving and diagnosing the evaluator; agreement supports a limited common orientation-failure observation, not its training cause or a Gaussian-specific claim.',
        selection='First32 by SHA256(R48-rotation-check-v1:object_id); independent of predictions and labels.',
        roles=list(ROLES), seed=11, replay_objects=32, rotation='Rz45 @ Rx30',
        coordinate_warning=2e-6, logit_warning=1e-3,
        post_result_diagnostic=True, training=False, angle_sweep=False,
        official_test_access=False, fresh_confirmation=False,
        changes_main_utility_decision=False, new_promotion_gate=False,
        device='cpu', threads=4, source_sha256=sha(__file__), inputs=inputs,
        original_frozen_source_count=len(lock['sources']), frozen_unix=time.time())
    write(DEST/'plan.json', plan)
    (DEST/'plan.sha256').write_text(sha(DEST/'plan.json')+'\n')

    configure()
    data = load('dev_eval', OUT)
    original_hashes = {k: hash_array(data[k]) for k in ('points', 'ids', 'labels')}
    base, tilted = evaluation_points(data, 'base'), evaluation_points(data, 'so3')
    reference = coordinate_reference(base)
    fixture = np.zeros((1, 2048, 3), dtype=np.float32)
    fixture[0, :3] = np.eye(3)
    bundle = dict(points=fixture, canonical=np.arange(2048)[None], ids=np.array(['fixture']))
    # Probe the actual transform with basis vectors, rather than copying its matrix.
    matrix = evaluation_points(bundle, 'so3')[0, :3].astype(np.float64).T
    expected_matrix = coordinate_reference(np.eye(3)).T
    recovered = tilted.astype(np.float64)@matrix
    distances = np.linalg.norm(base[:, :256].astype(np.float64)-base[:, 256:], axis=-1)
    tilted_distances = np.linalg.norm(tilted[:, :256].astype(np.float64)-tilted[:, 256:], axis=-1)
    geometry = dict(
        objects=len(base), points_per_object=base.shape[1], matrix=matrix.tolist(),
        matrix_reference_error=float(abs(matrix-expected_matrix).max()),
        orthogonality_error=float(abs(matrix.T@matrix-np.eye(3)).max()),
        determinant=float(np.linalg.det(matrix)),
        rotation_angle_degrees=float(np.degrees(np.arccos(np.clip((np.trace(matrix)-1)/2, -1, 1)))),
        up_axis_tilt_degrees=float(np.degrees(np.arccos(np.clip(matrix[1, 1], -1, 1)))),
        coordinate_reference_error=float(abs(tilted-reference).max()),
        inverse_coordinate_error=float(abs(recovered-base).max()),
        point_radius_error=float(abs(np.linalg.norm(tilted.astype(np.float64), axis=-1)-np.linalg.norm(base.astype(np.float64), axis=-1)).max()),
        paired_distance_error=float(abs(tilted_distances-distances).max()),
        source_arrays_unchanged=all(hash_array(data[k]) == digest for k, digest in original_hashes.items()))
    order = sorted(range(len(base)), key=lambda i: hashlib.sha256(
        ('R48-rotation-check-v1:'+str(data['ids'][i])).encode()).digest())[:32]
    replay = []
    for role in ROLES:
        path = OUT/'runs'/role/'11'
        state = torch.load(path/'best.pt', weights_only=True, map_location='cpu')['state_dict']
        model = make_model(role).eval()
        model.load_state_dict(state)
        base_logits = prediction(model, base[order])
        tilt_logits = prediction(model, reference[order])
        inverse_logits = prediction(model, recovered[order])
        with np.load(path/'dev_eval.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], data['ids']) or not np.array_equal(saved['labels'], data['labels']):
                raise ValueError('prediction identity mismatch')
            row = dict(role=role, seed=11,
                base_logit_error=float(abs(base_logits-saved['base'][order]).max()),
                tilt_logit_error=float(abs(tilt_logits-saved['so3'][order]).max()),
                inverse_base_logit_error=float(abs(inverse_logits-base_logits).max()),
                base_winner_changes=int(np.sum(base_logits.argmax(-1) != saved['base'][order].argmax(-1))),
                tilt_winner_changes=int(np.sum(tilt_logits.argmax(-1) != saved['so3'][order].argmax(-1))),
                inverse_base_winner_changes=int(np.sum(inverse_logits.argmax(-1) != base_logits.argmax(-1))))
        replay.append(row)
        print('REPLAY', row, flush=True)
    warnings = []
    for key in ('matrix_reference_error', 'orthogonality_error', 'coordinate_reference_error',
                'inverse_coordinate_error', 'point_radius_error', 'paired_distance_error'):
        if geometry[key] > plan['coordinate_warning']:
            warnings.append(key)
    if abs(geometry['determinant']-1) > plan['coordinate_warning'] or not geometry['source_arrays_unchanged']:
        warnings.append('matrix determinant or source mutation')
    for row in replay:
        if any(row[k] > plan['logit_warning'] for k in ('base_logit_error', 'tilt_logit_error', 'inverse_base_logit_error')):
            warnings.append(row['role']+' logits')
        if any(row[k] for k in ('base_winner_changes', 'tilt_winner_changes', 'inverse_base_winner_changes')):
            warnings.append(row['role']+' class changes')
    checked(OUT)
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('diagnostic input changed')
    write(DEST/'result.json', dict(geometry=geometry, replay=replay, warnings=warnings,
        selected_ids=data['ids'][order].tolist(), seconds=time.monotonic()-started,
        plan_sha256=sha(DEST/'plan.json'), source_sha256=sha(__file__),
        scope='Input transform checked on all old development objects; prediction replay only32 objects x3 models. No new rotation accuracy estimate or causal training diagnosis.',
        official_test_access=False, fresh_confirmation=False, goal_complete=False))
    print('RESULT', DEST/'result.json', 'warnings', warnings, flush=True)


if __name__ == '__main__':
    main()
