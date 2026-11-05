"""Native-cache inference and source-reuse reporting; no download or fitting."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from experiments import round65_run as runner
from experiments.round65_audit import independent_metrics
from experiments.round65_common import CONFIG, VARIANTS, sha
from experiments.native_stream_cache import schema, atomic_json
from experiments.native_memmap_runtime import install
from models.round65_actual_gs import NativeGSClassifier, ROLES as ORIGINAL_ROLES
from models.round73_native_relative5 import CompactNativeClassifier
from models.native_shape_attribution import NativeFixedIsotropic


def load_cached_population(directory, rows, expected_manifest_sha256):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('cache manifest identity changed')
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads((directory / 'identity.json').read_text())
    identity = {k: v for k, v in marker.items() if k != 'sha256'}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if marker['sha256'] != identity_sha or manifest['identity_sha256'] != identity_sha:
        raise ValueError('cache population identity changed')
    ids = [r['file_path'] for r in rows]
    if not manifest['complete'] or manifest['count'] != len(rows) or manifest['ids'] != ids:
        raise ValueError('incomplete or reordered population')
    if marker['rows'] != rows or len(ids) != len(set(ids)):
        raise ValueError('cache catalog differs from declared population')
    if marker['tokens'] != CONFIG['cache_tokens'] or marker['covariance_floor'] != CONFIG['covariance_floor']:
        raise ValueError('cache geometry configuration differs')
    if marker['sample_salt'] != CONFIG['sample_salt']:
        raise ValueError('cache sampling configuration differs')
    arrays = {}
    for name, (dtype, shape) in schema(len(rows), CONFIG['cache_tokens']).items():
        path = directory / (name + '.npy')
        if sha(path) != manifest['array_sha256'][path.name]:
            raise ValueError('cache array changed: ' + name)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError('cache array schema differs: ' + name)
        arrays[name] = value
    if not np.array_equal(arrays['labels'], [r['taxonomy_id'] for r in rows]):
        raise ValueError('labels and declared catalog differ')
    arrays.update(ids=np.asarray(ids), cache_identity_sha256=identity_sha,
                  cache_manifest_sha256=expected_manifest_sha256)
    return arrays


def population_groups(rows, historical_membership=None):
    ids = [r['file_path'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate population IDs')
    source = np.asarray([r['source'] for r in rows])
    if not set(source.tolist()) <= {'MACGS', 'ModelNet'}:
        raise ValueError('unknown source')
    groups = dict(full_population=np.ones(len(rows), dtype=bool),
                  MACGS=source == 'MACGS', ModelNet=source == 'ModelNet')
    if historical_membership is not None:
        lookup = {r['file_path']: r for r in historical_membership}
        if set(lookup) != set(ids) or len(lookup) != len(historical_membership):
            raise ValueError('historical membership must exactly cover the declared population')
        for row in rows:
            item = lookup[row['file_path']]
            if (item['taxonomy_id'], item['source'], item['model_id']) != (row['taxonomy_id'], row['source'], row['model_id']):
                raise ValueError('source identity mismatch')
        status = np.asarray([lookup[p]['historical_status'] for p in ids])
        known = {'used_original_train', 'original_test_not_used_in_R48_R58', 'not_ModelNet'}
        if not set(status.tolist()) <= known:
            raise ValueError('unresolved historical membership')
        reused = status == 'used_original_train'
        groups.update(historical_source_reused=reused, no_known_historical_source_overlap=~reused)
    # Empty source strata are represented explicitly, never as a successful score.
    return groups


def summarize_logits(logits, labels, groups):
    x, y = np.asarray(logits), np.asarray(labels)
    if x.shape != (len(y), 60) or not np.isfinite(x).all() or np.any((y < 0) | (y >= 60)):
        raise ValueError('invalid native classification output')
    result = {}
    for name, mask in groups.items():
        mask = np.asarray(mask)
        if mask.shape != y.shape or mask.dtype != np.bool_:
            raise ValueError('population mask is not aligned')
        count = int(mask.sum())
        classes = sorted(set(y[mask].tolist()))
        result[name] = dict(count=count, classes=classes, class_count=len(classes),
                            metrics=independent_metrics(x[mask], y[mask]) if count else None)
    return result


def make_model(role):
    if role == 'relative5':
        return CompactNativeClassifier(role)
    if role == 'fixed_isotropic5':
        return NativeFixedIsotropic(role)
    if role in ORIGINAL_ROLES:
        return NativeGSClassifier(role)
    raise ValueError('undeclared model role: ' + role)


@torch.inference_mode()
def export_one(bundle, checkpoint, variant, groups, destination):
    """Caller must seal model/population identities before supplying test inputs."""
    if variant not in VARIANTS:
        raise ValueError('undeclared evaluation condition')
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('selected checkpoint changed')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_path = destination.with_suffix('.json')
    group_identity = {k: hashlib.sha256(np.asarray(v, dtype=np.bool_).tobytes()).hexdigest()
                      for k, v in groups.items()}
    identity = dict(checkpoint=checkpoint, variant=variant, cache_identity_sha256=bundle['cache_identity_sha256'],
        cache_manifest_sha256=bundle['cache_manifest_sha256'], runtime_sha256=sha(__file__), group_identity=group_identity)
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old['identity'] != identity or sha(destination) != old['prediction_sha256']:
            raise ValueError('existing export has different identity')
        return old
    if destination.exists() or destination.with_suffix('.pending').exists():
        raise ValueError('preserve interrupted export for explicit recovery')
    runner.configure()
    install(runner)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one selected CUDA device required')
    payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    if (payload['role'], payload['seed'], payload['epoch']) != (checkpoint['role'], checkpoint['seed'], checkpoint['selected_epoch']):
        raise ValueError('checkpoint metadata differs')
    model = make_model(checkpoint['role']).cuda().eval()
    model.load_state_dict(payload['state_dict'])
    logits, original_metrics = runner.evaluate(model, bundle, variant)
    summary = summarize_logits(logits, bundle['labels'], groups)
    all_metrics = independent_metrics(logits, bundle['labels'])
    difference = max(abs(all_metrics[k] - original_metrics[k]) for k in ('mca', 'oa', 'ce'))
    if difference > 1e-8:
        raise ValueError('independent metric disagreement')
    temporary = destination.with_suffix('.pending')
    with temporary.open('xb') as stream:
        np.savez(stream, logits=logits, ids=bundle['ids'], labels=bundle['labels'])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    record = dict(identity=identity, prediction_sha256=sha(destination), groups=summary,
        independent_metric_difference=difference, optimizer_updates=0)
    atomic_json(record_path, record)
    return record
