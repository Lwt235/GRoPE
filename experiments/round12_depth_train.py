"""Locked, paired Round12 depth development; never loads historical test data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from models.round12_depth_attention import DepthAttention, FUSION_ROLES, ROLES, depth_loss
from models.round7_depth_attention import FIELDS
from experiments.round7_depth_train import load_data, batch, evaluate, metric_rows, macro


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
BASE_ROOTS = (ROOT/'results/round7_depth_old', ROOT/'results/round7_depth_bm4')
SCENES = {'train': ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1'),
          'val': ('brown_bm_4/brown_bm_4', 'harvard_c8/hv_c8_3')}
SOURCES = ('ROUND12_DEPTH_PROTOCOL.md', 'models/round12_depth_attention.py',
           'experiments/round12_depth_train.py', 'experiments/round12_depth_analysis.py',
           'models/round7_depth_attention.py', 'models/gaussian_quadratic_pe.py',
           'experiments/round7_depth_train.py', 'experiments/round7_depth_data.py',
           'experiments/round6_region_data.py', 'experiments/round5_rgbd_data.py',
           'tests/test_round12_depth_attention.py', 'tests/test_round12_depth_analysis.py',
           'tests/test_round12_depth_train.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sources_now():
    return {name: sha(ROOT/name) for name in SOURCES}


def original_manifests():
    """Strict allowlist: only the five old development scenes and their files."""
    result = {}
    for split, scenes in SCENES.items():
        rows = []
        for root in BASE_ROOTS:
            for directory in sorted((root/split).glob('*')):
                manifest_path = directory/'manifest.json'
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest['split'] != split or manifest['scene'] not in scenes:
                    raise ValueError('unexpected original development scene')
                expected = 192 if split == 'train' else 64
                pairs = [tuple(pair) for pair in manifest['pairs']]
                if len(pairs) != expected or len(set(pairs)) != expected:
                    raise ValueError('missing/duplicate original pair')
                for name, field in (('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')):
                    if sha(directory/name) != manifest[field]:
                        raise ValueError('changed original '+name)
                rows.append(dict(scene=manifest['scene'], path=str(directory.resolve()),
                                 manifest_sha256=sha(manifest_path), pairs=manifest['pairs'],
                                 input_sha256=manifest['input_sha256'],
                                 label_sha256=manifest['label_sha256']))
        if sorted(row['scene'] for row in rows) != sorted(scenes):
            raise ValueError('incomplete/duplicate original scenes')
        result[split] = rows
    return result


def geometry_inventory(geometry_root, original):
    """Freeze every role hash; no role arrays or sensor labels are opened here."""
    from experiments import round12_depth_geometry as producer
    geometry_root = Path(geometry_root).resolve()
    complete = geometry_root/'completion.json'
    export_lock = geometry_root/'lock.json'
    if not complete.is_file() or not export_lock.is_file():
        raise ValueError('geometry export is not complete')
    producer.verify_lock(geometry_root, inputs=True)
    producer_lock = json.loads(export_lock.read_text())
    preflight_path = geometry_root/'preflight.json'
    preflight = json.loads(preflight_path.read_text())
    completion = json.loads(complete.read_text())
    if (preflight.get('passed') is not True or preflight['lock_sha256'] != sha(export_lock)
            or completion['lock_sha256'] != sha(export_lock)
            or completion['preflight_sha256'] != sha(preflight_path)
            or completion['counts'] != {'train': 576, 'val': 128}
            or completion['role_count'] != 8 or completion['geometry_contains_content'] is not False
            or completion['fresh_test_read'] is not False):
        raise ValueError('geometry preflight/completion chain mismatch')
    completed_manifests = completion['manifests']
    expected_scenes = {(row['scene'], split) for split, rows in original.items() for row in rows}
    if (len(completed_manifests) != 5
            or {(row['scene'], row['split']) for row in completed_manifests} != expected_scenes):
        raise ValueError('geometry completion scene set mismatch')
    for row in completed_manifests:
        if row['path'] != f"{row['split']}/{row['scene'].replace('/', '__')}/manifest.json":
            raise ValueError('geometry completion path mismatch')
        if sha(geometry_root/row['path']) != row['sha256']:
            raise ValueError('geometry completion manifest hash mismatch')
    digest_file = geometry_root/'lock.sha256'
    if digest_file.exists() and digest_file.read_text().strip().split()[0] != sha(export_lock):
        raise ValueError('geometry lock digest mismatch')
    records, configs = {}, {}
    for split, base_rows in original.items():
        records[split] = []
        for base in base_rows:
            directory = geometry_root/split/base['scene'].replace('/', '__')
            manifest_path = directory/'manifest.json'
            m = json.loads(manifest_path.read_text())
            if m['lock_sha256'] != sha(export_lock) or m['preflight_sha256'] != sha(preflight_path):
                raise ValueError('geometry scene does not belong to completed export')
            if (m['scene'] != base['scene'] or m['split'] != split or m['pairs'] != base['pairs']
                    or Path(m['r7_root']).resolve() != Path(base['path']).resolve()):
                raise ValueError('geometry/original pair order or path mismatch')
            for own, original_key in (('r7_manifest_sha256', 'manifest_sha256'),
                                      ('r7_input_sha256', 'input_sha256'),
                                      ('r7_label_sha256', 'label_sha256')):
                if m[own] != base[original_key]:
                    raise ValueError('geometry/original provenance mismatch')
            if set(m['roles']) != set(FUSION_ROLES):
                raise ValueError('incomplete geometry roles')
            role_files = {}
            for role in FUSION_ROLES:
                entry = m['roles'][role]
                if entry['path'] != role+'.npy' or entry['dtype'] != 'float32':
                    raise ValueError('geometry path/dtype contract mismatch')
                if entry['shape'] != [len(base['pairs']), 432, 432]:
                    raise ValueError('geometry shape contract mismatch')
                path = directory/entry['path']
                if sha(path) != entry['sha256']:
                    raise ValueError('geometry array hash mismatch')
                config = {key: entry[key] for key in ('method', 'h', 'temperature')}
                if role in configs and configs[role] != config:
                    raise ValueError('geometry configuration varies by scene')
                configs[role] = config
                role_files[role] = dict(path=str(path), sha256=entry['sha256'], **config)
            records[split].append(dict(scene=base['scene'], path=str(directory),
                                       manifest_sha256=sha(manifest_path), roles=role_files))
    return dict(root=str(geometry_root), completion_sha256=sha(complete),
                lock_sha256=sha(export_lock), preflight_sha256=sha(preflight_path),
                scenes=records, configurations=configs)


def stress_authorization(stress_root, audit_path, verify_artifacts=True):
    """Hard promotion dependency, verified before a depth lock or run exists."""
    root, audit_path = Path(stress_root).resolve(), Path(audit_path).resolve()
    paths = {name: root/name for name in ('lock.json', 'summary.json', 'completion.json')}
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    hashes = {name: sha(path) for name, path in paths.items()}
    if (root/'lock.sha256').read_text().strip() != hashes['lock.json']:
        raise ValueError('stress source lock digest mismatch')
    if (values['summary.json']['lock_sha256'] != hashes['lock.json']
            or values['completion.json']['lock_sha256'] != hashes['lock.json']
            or values['completion.json']['fresh_test_read'] is not False
            or values['summary.json']['gates']['passed'] is not True):
        raise ValueError('stress gate failed or result provenance mismatched')
    completed_files = values['completion.json']['files']
    if completed_files.get('summary.json') != hashes['summary.json']:
        raise ValueError('stress completion does not bind the passing summary')
    if verify_artifacts:
        for name, digest in completed_files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(root/name) != digest:
                raise ValueError('stress completed artifact changed')
        for name, digest in values['lock.json']['source_hashes'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('stress frozen source changed')
    audit = json.loads(audit_path.read_text())
    if (audit.get('passed') is not True or audit.get('stress_lock_sha256') != hashes['lock.json']
            or audit.get('stress_summary_sha256') != hashes['summary.json']
            or audit.get('stress_completion_sha256') != hashes['completion.json']):
        raise ValueError('independent stress audit missing, failed or unbound')
    return dict(root=str(root), audit_path=str(audit_path), audit_sha256=sha(audit_path),
                artifacts_sha256=hashes, stress_passed=True, independent_audit_passed=True)


def freeze(out, geometry_root, stress_root, stress_audit):
    out = Path(out)
    if out.exists():
        raise ValueError('refuse existing training directory')
    promotion = stress_authorization(stress_root, stress_audit)
    original = original_manifests()
    geometry = geometry_inventory(geometry_root, original)
    mapping = sources_now()
    out.mkdir(parents=True)
    for name in mapping:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    lock = dict(version=1, source_hashes=mapping, roles=list(ROLES), seeds=list(SEEDS),
                steps=1000, batch_size=8, old_data=original, geometry=geometry,
                test_access_authorized=False, stress_promotion=promotion,
                scope='old development only; source-value utility and measured compression cost')
    write_json(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN', sha(out/'lock.json'), flush=True)


def verify_lock(out, verify_data=False):
    out = Path(out)
    lock = json.loads((out/'lock.json').read_text())
    if (out/'lock.sha256').read_text().strip() != sha(out/'lock.json'):
        raise ValueError('training lock changed')
    if (lock['source_hashes'] != sources_now() or lock['roles'] != list(ROLES)
            or lock['seeds'] != list(SEEDS) or lock['steps'] != 1000
            or lock['test_access_authorized'] is not False):
        raise ValueError('training source/contract changed')
    promotion = lock['stress_promotion']
    if stress_authorization(promotion['root'], promotion['audit_path'], verify_artifacts=verify_data) != promotion:
        raise ValueError('stress promotion evidence changed after depth freeze')
    g = lock['geometry']; root = Path(g['root'])
    if (sha(root/'lock.json') != g['lock_sha256'] or sha(root/'completion.json') != g['completion_sha256']
            or sha(root/'preflight.json') != g['preflight_sha256']):
        raise ValueError('geometry completion or source lock changed')
    for name, digest in json.loads((root/'lock.json').read_text())['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('geometry producer source changed')
    if verify_data and original_manifests() != lock['old_data']:
        raise ValueError('old input/label population changed')
    return lock


def load_role_data(role, split, lock, device='cpu'):
    if role not in ROLES or split not in SCENES:
        raise ValueError('only declared development role/split permitted')
    data, ids, sources = load_data(BASE_ROOTS, split, device)
    expected = [(row['scene'], q, k) for row in lock['old_data'][split] for q, k in row['pairs']]
    if [(row['scene'], row['q_index'], row['k_index']) for row in ids] != expected:
        raise ValueError('original loader pair ordering changed')
    if role != 'singleview':
        arrays = []
        for row in lock['geometry']['scenes'][split]:
            if sha(Path(row['path'])/'manifest.json') != row['manifest_sha256']:
                raise ValueError('geometry scene manifest changed')
            entry = row['roles'][role]; path = Path(entry['path'])
            if sha(path) != entry['sha256']:
                raise ValueError('selected role geometry changed')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            count = sum(item['scene'] == row['scene'] for item in ids)
            if array.dtype != np.float32 or array.shape != (count, 432, 432) or not np.isfinite(array).all():
                raise ValueError('selected geometry shape/dtype/finite failure')
            arrays.append(array)
        data['geometry_logits'] = torch.from_numpy(np.concatenate(arrays)).to(device)
    return data, ids, sources


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit(role, seed, train, train_ids, val, val_ids, out, lock):
    dest = Path(out)/role/str(seed)
    dest.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    temperature = 5. if role == 'singleview' else lock['geometry']['configurations'][role]['temperature']
    model = DepthAttention(role, temperature).to(train['q'].device)
    if sum(p.numel() for p in model.parameters()) != 41345:
        raise ValueError('model parameter capacity changed')
    initial_sha = parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed); sequence = hashlib.sha256()
    started = time.time(); history = []; training_losses = []; best = float('inf')
    first_gradients = None
    if train['q'].is_cuda:
        torch.cuda.reset_peak_memory_stats(train['q'].device)
    for step in range(1001):
        if step % 100 == 0:
            result, _ = evaluate(model, val, val_ids)
            score = result['summary']['metrics']['AbsRel']
            history.append(dict(step=step, validation=result['summary']))
            if score < best:
                best = score; best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print('VAL', role, seed, step, round(score, 7), flush=True)
        if step == 1000:
            break
        model.train(); indices = rng.integers(len(train_ids), size=8)
        sequence.update(indices.astype('<i8').tobytes())
        d = batch(train, indices); optimizer.zero_grad(set_to_none=True)
        prediction = model(d); loss = depth_loss(prediction, d['target'], d['valid'])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite depth loss')
        loss.backward()
        if step == 0:
            first_gradients = {name: dict(present=p.grad is not None,
                              finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),
                              l1=float(p.grad.abs().sum()) if p.grad is not None else None)
                               for name, p in model.named_parameters()}
            if not all(row['present'] and row['finite'] for row in first_gradients.values()):
                raise ValueError('missing/nonfinite actual training gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); training_losses.append(float(loss.detach()))
    model.load_state_dict(best_state)
    result, prediction = evaluate(model, val, val_ids)
    saved = dict(depth=prediction.detach().cpu().numpy())
    interventions = {}
    for mode in ('zero_v', 'reverse_v'):
        changed_result, changed = evaluate(model, val, val_ids, mode)
        saved[mode] = changed.detach().cpu().numpy()
        interventions[mode] = dict(**changed_result,
            mean_absolute_output_change_m=float((changed-prediction).abs().mean()))
    torch.save(dict(role=role, seed=seed, step=best_step, temperature=temperature,
                    state_dict=best_state, source_hashes=lock['source_hashes']), dest/'best.pt')
    np.savez_compressed(dest/'val_predictions.npz', **saved)
    write_json(dest/'validation.json', result)
    write_json(dest/'interventions.json', interventions)
    metadata = dict(role=role, seed=seed, steps=1000, best_step=best_step,
        best_validation_AbsRel=best, temperature=temperature, parameter_count=41345,
        source_hashes=lock['source_hashes'], lock_sha256=sha(Path(out)/'lock.json'),
        initialization_sha256=initial_sha, batch_sequence_sha256=sequence.hexdigest(),
        checkpoint_sha256=sha(dest/'best.pt'), prediction_sha256=sha(dest/'val_predictions.npz'),
        validation_sha256=sha(dest/'validation.json'), interventions_sha256=sha(dest/'interventions.json'),
        history=history, training_losses=training_losses, first_step_gradients=first_gradients,
        seconds=time.time()-started, device=str(train['q'].device),
        GPU=torch.cuda.get_device_name(train['q'].device) if train['q'].is_cuda else None,
        max_memory_allocated=torch.cuda.max_memory_allocated(train['q'].device) if train['q'].is_cuda else None,
        geometry_precomputed=True, training_seconds_are_not_complete_pipeline_latency=True,
        torch_version=torch.__version__, numpy_version=np.__version__,
        original_train_pairs=len(train_ids), original_val_pairs=len(val_ids),
        independent_confirmation=False)
    verify_lock(out, verify_data=True)
    if role != 'singleview':
        for split in ('train', 'val'):
            for row in lock['geometry']['scenes'][split]:
                entry = row['roles'][role]
                if sha(entry['path']) != entry['sha256']:
                    raise ValueError('geometry changed during training')
    write_json(dest/'run.json', metadata)
    print('DONE', role, seed, best_step, best, flush=True)


@torch.inference_mode()
def paired_timing(out, device):
    """Rebuild geometry from common projected pixels, then actual learned V/head."""
    from experiments import round12_depth_geometry as producer
    out = Path(out); lock = verify_lock(out, verify_data=True)
    if (out/'timing.json').exists():
        raise ValueError('refuse timing overwrite')
    root = Path(lock['geometry']['root'])
    producer.verify_lock(root)
    device = torch.device(device)
    all_scenes = sorted((row for rows in lock['old_data'].values() for row in rows), key=lambda row: row['scene'])
    records, common_times, run_hashes = [], [], {}
    for scene_index, record in enumerate(all_scenes):
        scene = record['scene']; directory = Path(record['path'])
        with np.load(directory/'inputs.npz', allow_pickle=False) as arrays:
            inputs = {name: torch.from_numpy(arrays[name][:1].copy()).to(device) for name in FIELDS}
        caches, metadata, cache_hashes, _ = producer.load_scene_cache(scene)
        producer_lock = json.loads((root/'lock.json').read_text())
        if any(producer_lock['input_files'].get(name) != digest for name, digest in cache_hashes.items()):
            raise ValueError('timing cache source changed')
        q_index, k_index = metadata['pairs'][0]
        q_cache, k_cache = caches[q_index], caches[k_index]
        producer.prepare_forward_observation(q_cache, k_cache, 'cpu')
        cpu_samples = []
        for _ in range(15):
            started = time.perf_counter()
            observation_cpu = producer.prepare_forward_observation(q_cache, k_cache, 'cpu')
            cpu_samples.append(1000*(time.perf_counter()-started))
        observation = {name: value.to(device) for name, value in observation_cpu.items()}
        common_times.append(dict(scene=scene, pair=[q_index, k_index], samples_ms=cpu_samples,
                                 median_ms=float(np.median(cpu_samples)), warmups=1,
                                 includes='common CPU dense_observations projection and tensor wrapping',
                                 excludes='RGB/depth prediction, cache IO, GPU transfer'))
        del caches, q_cache, k_cache
        for seed_index, seed in enumerate(SEEDS):
            offset = (scene_index+seed_index) % len(ROLES)
            order = ROLES[offset:]+ROLES[:offset]
            for role in order:
                folder = out/role/str(seed); run_path = folder/'run.json'
                run = json.loads(run_path.read_text()); run_hashes[str(run_path)] = sha(run_path)
                if run['lock_sha256'] != sha(out/'lock.json') or sha(folder/'best.pt') != run['checkpoint_sha256']:
                    raise ValueError('timing checkpoint/lock mismatch')
                config = None if role == 'singleview' else lock['geometry']['configurations'][role]
                temperature = 5. if config is None else config['temperature']
                model = DepthAttention(role, temperature).to(device)
                checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=False)
                if checkpoint['role'] != role or checkpoint['seed'] != seed or checkpoint['step'] != run['best_step']:
                    raise ValueError('timing checkpoint identity mismatch')
                model.load_state_dict(checkpoint['state_dict']); model.eval()

                def forward():
                    if role == 'singleview':
                        return model(inputs)
                    geometry = producer.compute_geometry(observation, role, config).float()[None]
                    return model(dict(inputs, geometry_logits=geometry))

                for _ in range(3):
                    forward()
                samples = []
                for _ in range(15):
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    started = time.perf_counter(); prediction = forward()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    samples.append(1000*(time.perf_counter()-started))
                if not torch.isfinite(prediction).all():
                    raise ValueError('timing produced nonfinite depth')
                records.append(dict(role=role, seed=seed, scene=scene, pair=[q_index, k_index],
                    warmups=3, samples_ms=samples, median_ms=float(np.median(samples)),
                    checkpoint_sha256=run['checkpoint_sha256']))
                print('TIMING', scene, role, seed, round(records[-1]['median_ms'], 3), flush=True)
        del observation, inputs
    verify_lock(out, verify_data=True)
    for path, digest in run_hashes.items():
        if sha(path) != digest:
            raise ValueError('selected training record changed during timing')
    result = dict(lock_sha256=sha(out/'lock.json'), geometry_recomputed=True, rows=records,
        mean_median_ms={role:float(np.mean([row['median_ms'] for row in records if row['role']==role])) for role in ROLES},
        common_dense_projection_cpu=common_times, run_hashes=run_hashes,
        geometry_export_completion=json.loads((root/'completion.json').read_text()),
        includes='support preparation, rectangle integral, float32 cast, learned QK, attention, V/head',
        excludes='common RGB/depth predictors, dense camera projection, cache IO, transfers',
        role_order='cyclic by scene_index+seed_index', device=str(device),
        GPU=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        entire_vision_system_latency=False)
    write_json(out/'timing.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'train', 'timing'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--geometry-root', type=Path, default=ROOT/'results/round12_depth_geometry')
    parser.add_argument('--stress-root', type=Path, default=ROOT/'results/round12_stress')
    parser.add_argument('--stress-audit', type=Path)
    parser.add_argument('--roles', nargs='+', choices=ROLES, default=list(ROLES))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.action == 'freeze':
        if args.stress_audit is None:
            parser.error('freeze requires --stress-audit with passing independent bound audit')
        freeze(args.out, args.geometry_root, args.stress_root, args.stress_audit); return
    if args.action == 'timing':
        paired_timing(args.out, args.device); return
    lock = verify_lock(args.out, verify_data=True)
    for role in args.roles:
        train, ti, _ = load_role_data(role, 'train', lock, args.device)
        val, vi, _ = load_role_data(role, 'val', lock, args.device)
        for seed in args.seeds:
            fit(role, seed, train, ti, val, vi, args.out, lock)
        del train, val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
