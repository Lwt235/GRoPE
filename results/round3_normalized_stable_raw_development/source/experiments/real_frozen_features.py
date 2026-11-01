"""Export frozen ImageNet ResNet18 regional features for development only.

Only train/validation pairs from the two declared adjacent-camera scenes may
be opened. Pair geometry, validity and labels are copied without modification.
No weights are downloaded, no learning is run, and no target statistics define
the fixed orthogonal projection. The official torchvision model is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


SCENES = ('teatime', 'waldo_kitchen')
SPLITS = ('train', 'validation')
PROJECTION_SEED = 20261030
IMAGE_SIZE = (192, 144)
GRID = (8, 6)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fixed_projection(input_width, output_width=108, seed=PROJECTION_SEED):
    """Label-free QR projection; 64->108 is isometric, 128->108 is truncation."""
    generator = torch.Generator().manual_seed(seed+input_width)
    bigger, smaller = max(input_width, output_width), min(input_width, output_width)
    normal = torch.randn(bigger, smaller, generator=generator, dtype=torch.float64)
    orthogonal, triangular = torch.linalg.qr(normal, mode='reduced')
    # Fix QR column signs so an implementation convention cannot change them.
    sign = torch.where(triangular.diagonal() < 0, -1., 1.)
    orthogonal = orthogonal*sign
    return (orthogonal if input_width >= output_width else orthogonal.T).float()


def project_features(pooled, projection):
    return F.normalize(torch.as_tensor(pooled, dtype=torch.float32) @ projection,
                       dim=-1, eps=1e-12).numpy()


def official_backbone(weights_path):
    """Strictly load the existing checkpoint into torchvision's own ResNet18."""
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.models.resnet import BasicBlock
    weights_path = Path(weights_path)
    weight_hash = sha256(weights_path)
    if not weight_hash.startswith('f37072fd'):
        raise ValueError('local checkpoint does not match the official ResNet18 V1 hash prefix')
    model = resnet18(weights=None)
    result = model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('strict official architecture load failed')
    model.cpu().eval().requires_grad_(False)
    source = Path(inspect.getsourcefile(BasicBlock))
    metadata = dict(architecture='torchvision.models.resnet18', torchvision_version=torchvision.__version__,
                    torch_version=torch.__version__, weights='ResNet18_Weights.IMAGENET1K_V1',
                    weights_path=str(weights_path.resolve()), weights_sha256=weight_hash,
                    official_weights_url=ResNet18_Weights.IMAGENET1K_V1.url,
                    original_pretraining='supervised ImageNet-1K classification; not correspondence training',
                    official_resnet_source=str(source), official_resnet_source_sha256=sha256(source),
                    strict_load=True, missing_keys=[], unexpected_keys=[], frozen=True, device='cpu')
    return model, metadata


def preprocess_image(path):
    with Image.open(path) as source:
        original_size = source.size
        # Full-image resize keeps exactly the existing matching grid. The
        # official 224 center-crop classification transform is not appropriate.
        rgb = np.asarray(source.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BOX)).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()/255.
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor-mean)/std, dict(original_size=list(original_size), resized_rgb_sha256=array_sha256(rgb))


@torch.inference_mode()
def extract_pooled(path, model):
    image, metadata = preprocess_image(path)
    value = image[None]
    value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
    layer1 = model.layer1(value)
    layer2 = model.layer2(layer1)
    if layer1.shape != (1, 64, 36, 48) or layer2.shape != (1, 128, 18, 24):
        raise ValueError('unexpected official dense feature layout')
    features = {}
    for name, value in [('layer1', layer1), ('layer2', layer2)]:
        # 6x8 row-major grid. Layer2 pools 3x3 dense cells per region;
        # layer1 pools 6x6. Receptive fields can cross region boundaries.
        pooled = F.adaptive_avg_pool2d(value, (GRID[1], GRID[0]))
        features[name+'_mean'] = pooled[0].permute(1, 2, 0).reshape(48, value.shape[1]).numpy().copy()
    metadata['normalized_input_sha256'] = array_sha256(image.numpy())
    return features, metadata


def development_pairs(manifest):
    """Filter before any NPZ/image is opened; test files need not exist."""
    pairs = [copy.deepcopy(pair) for pair in manifest['pairs'] if pair['split'] in SPLITS]
    if not pairs:
        raise ValueError('scene has no permitted development pairs')
    return pairs


def export_scene(source, destination, model, model_metadata, projections, selected_layer, cache_root):
    source, destination, cache_root = Path(source), Path(destination), Path(cache_root)
    if source.resolve() == destination.resolve():
        raise ValueError('output must not overwrite the source dataset')
    original = json.loads((source/'manifest.json').read_text())
    if original['scene'] not in SCENES:
        raise ValueError('only declared development scenes may be opened')
    pairs = development_pairs(original)
    allowed_ids = {pair[side+'_id'] for pair in pairs for side in ('q', 'k')}
    dataset_path = Path(original['dataset'])
    dataset = json.loads(dataset_path.read_text())
    images = {image['id']: image for image in dataset['images'] if image['id'] in allowed_ids}
    if images.keys() != allowed_ids:
        raise ValueError('missing development image metadata')
    destination.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached, image_records, source_images = {}, [], {}
    for image_id in sorted(allowed_ids):
        image_path = (dataset_path.parent/images[image_id]['path']).resolve()
        image_hash = sha256(image_path)
        expected = original.get('source_images_sha256', {}).get(str(image_path))
        if expected is not None and image_hash != expected:
            raise ValueError('source photo hash changed')
        source_images[str(image_path)] = image_hash
        cache_file = cache_root/f'image_{image_id:06d}.npz'
        metadata_file = cache_file.with_suffix('.json')
        identity = dict(image_id=image_id, image_path=str(image_path), image_sha256=image_hash,
                        weights_sha256=model_metadata['weights_sha256'],
                        model_runtime={key:model_metadata.get(key) for key in
                                       ('architecture', 'torch_version', 'torchvision_version', 'official_resnet_source_sha256')},
                        projection_sha256={key:array_sha256(value.numpy()) for key,value in projections.items()},
                        image_size=list(IMAGE_SIZE), grid=list(GRID),
                        preprocessing='full RGB image BOX resize, ImageNet mean/std, no crop',
                        implementation_sha256=sha256(Path(__file__)))
        if cache_file.exists() and metadata_file.exists():
            record = json.loads(metadata_file.read_text())
            if record['identity'] != identity or sha256(cache_file) != record['cache_sha256']:
                raise ValueError('cached image features no longer match frozen extraction settings')
            with np.load(cache_file, allow_pickle=False) as arrays:
                feature_data = {key:arrays[key].copy() for key in arrays.files}
        else:
            feature_data, preparation = extract_pooled(image_path, model)
            for layer, projection in projections.items():
                feature_data[layer+'_projected'] = project_features(feature_data[layer+'_mean'], projection)
            np.savez_compressed(cache_file, **feature_data)
            record = dict(identity=identity, preparation=preparation, cache_path=str(cache_file.resolve()),
                          cache_sha256=sha256(cache_file),
                          feature_array_sha256={key:array_sha256(value) for key,value in feature_data.items()})
            metadata_file.write_text(json.dumps(record, indent=2)+'\n')
        chosen = feature_data[selected_layer+'_projected']
        if chosen.shape != (48, 108) or not np.isfinite(chosen).all():
            raise ValueError('invalid projected visual features')
        if not np.allclose(np.linalg.norm(chosen, axis=-1), 1, atol=2e-6):
            raise ValueError('visual features have zero or nonunit norm')
        cached[image_id] = chosen
        image_records.append(record)
    pair_records = []
    for pair in pairs:
        input_path = source/pair['file']
        input_hash = sha256(input_path)
        if pair['npz_sha256'] != input_hash:
            raise ValueError('source development pair hash changed')
        with np.load(input_path, allow_pickle=False) as arrays:
            original_arrays = {key:arrays[key].copy() for key in arrays.files}
        changed = dict(original_arrays)
        changed['q_features'] = cached[pair['q_id']]
        changed['k_features'] = cached[pair['k_id']]
        for key in original_arrays:
            if key not in ('q_features', 'k_features') and not np.array_equal(changed[key], original_arrays[key], equal_nan=True):
                raise ValueError('nonfeature source array changed')
        output_path = destination/pair['file']
        np.savez_compressed(output_path, **changed)
        pair['source_npz_sha256'] = input_hash
        pair['npz_sha256'] = sha256(output_path)
        pair['nonfeature_arrays_identical'] = True
        pair['nonfeature_array_sha256'] = {key:array_sha256(value) for key,value in original_arrays.items()
                                          if key not in ('q_features', 'k_features')}
        pair_records.append(pair)
    result = copy.deepcopy(original)
    result.update(pairs=pair_records, pair_count=len(pair_records), used_views=len(allowed_ids),
                  available_frames=len(allowed_ids), source_available_frames=original.get('available_frames'),
                  frames=[frame for frame in original['frames'] if frame['image_id'] in allowed_ids],
                  source_images_sha256=source_images, feature_shape=[48,108],
                  source_manifest_sha256=sha256(source/'manifest.json'),
                  source_manifest=str((source/'manifest.json').resolve()),
                  feature_preparation=dict(layer=selected_layer, pooling='dense regional average over 6x8 grid',
                                           projection='fixed Gaussian QR, independent of all data/labels',
                                           projection_seed=PROJECTION_SEED, output_l2_normalized=True,
                                           model=model_metadata, image_features=image_records,
                                           exported_splits=list(SPLITS), test_images_or_npz_opened=False,
                                           frozen_geometry_labels_and_masks=True))
    (destination/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    model, metadata = official_backbone(args.weights)
    projections = {'layer1':fixed_projection(64), 'layer2':fixed_projection(128)}
    args.out.mkdir(parents=True, exist_ok=True)
    projection_records = {}
    for layer, projection in projections.items():
        path = args.out/f'{layer}_projection.npy'
        np.save(path, projection.numpy())
        projection_records[layer] = dict(path=str(path.resolve()), sha256=sha256(path),
                                         array_sha256=array_sha256(projection.numpy()), shape=list(projection.shape))
    records = []
    for scene in SCENES:
        result = export_scene(args.source/scene, args.out/scene, model, metadata, projections,
                              args.layer, args.out/'image_cache'/scene)
        records.append(dict(scene=scene, pairs=result['pair_count'], images=result['used_views'],
                            splits={split:sum(pair['split']==split for pair in result['pairs']) for split in SPLITS},
                            manifest_sha256=sha256(args.out/scene/'manifest.json')))
        print(json.dumps(records[-1]), flush=True)
    summary = dict(config={key:str(value) for key,value in vars(args).items()}, model=metadata,
                   projections=projection_records, records=records, source_sha256=sha256(Path(__file__)),
                   elapsed_seconds=time.monotonic()-started, fitting_or_test_selection=False,
                   allowed_scenes=list(SCENES), allowed_splits=list(SPLITS))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    readme = f'''# Frozen ResNet18 visual features: development only

The official torchvision ResNet18 architecture strictly loaded local ImageNet-1K V1 weights.
Weights SHA256: `{metadata['weights_sha256']}`. No download, optimization or target-fitted projection was used.
Official checkpoint: {metadata['official_weights_url']}

Only teatime / waldo_kitchen train and validation pairs and their photos were opened. No test pair/photo or external scene was opened.
Each image is converted to RGB, BOX-resized as a full image to 192x144, and ImageNet mean/std normalized. No crop is used.
The current pair features are frozen **{args.layer}** dense average features over the existing row-major 8x6 region grid,
followed by a fixed seeded orthogonal projection to 108 dimensions and L2 normalization.
Layer2 uses 128->108 projection; layer1's optional 64->108 map is isometric before normalization.
Both layers' unprojected pooled vectors and projected alternatives are cached per image. The active layer is identical for every PE method.
Nonfeature pair arrays (geometry, cameras, labels, masks and track counts) are copied exactly and hashed individually.

This is supervised ImageNet classification pretraining, not a matching-trained visual backbone or a dense correspondence benchmark.
The official classifier's 256-resize / 224-center-crop evaluation recipe is deliberately not used: it would change the token/geometry grid.
Dense CNN receptive fields cross region boundaries; stride/padding make region pooling approximate image-region localization.
No claim about official ImageNet accuracy, stronger matching quality, or a universal positional encoding follows from this export.
The inherited sparse SfM target and reconstructed-GS limitations remain unchanged.

The cache records exact source-photo, normalized-input, feature-array, projection and weight hashes and can be reused only with identical settings.
These are development artifacts; a later confirmation export must freeze layer/projection/preprocessing first and explicitly authorize the additional splits.

Reproduce: `python -m experiments.real_frozen_features --source {args.source} --out {args.out} --layer {args.layer}`
'''
    (args.out/'README.md').write_text(readme)
    print('COMPLETE', summary['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/round3_real_support_data_adjacent'))
    parser.add_argument('--out', type=Path, default=Path('results/round3_frozen_visual_data'))
    parser.add_argument('--weights', type=Path, default=Path('/home/liwt/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    parser.add_argument('--layer', choices=['layer1','layer2'], default='layer2')
    main(parser.parse_args())
