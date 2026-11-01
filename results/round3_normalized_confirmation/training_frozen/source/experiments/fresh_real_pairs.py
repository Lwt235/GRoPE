"""Prepare untouched-camera confirmation data, without evaluating any matcher.

All previously exported posed-image IDs are excluded before opening new image
pixels or pair labels. Remaining images are filename-ordered and grouped by
threes, with the first querying the next two. Sparse-track statistics never
select pairs. The existing geometry exporter is wrapped, not edited.

Frozen ResNet18 feature copying is an explicit new confirmation-preparation
phase. It calls only the already frozen extraction primitives, rather than
bypassing the development-only restrictions in its old scene exporter.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments import real_support_data as base
from experiments import real_frozen_features as visual


SUBSET = 'previously_unexported_cameras'
SCENES = tuple(base.DEFAULT_SOURCES)


def ordered_pairs(remaining_ids):
    pairs = []
    for start in range(0, len(remaining_ids), 3):
        group = remaining_ids[start:start+3]
        if len(group) < 2:
            continue
        for offset, key in enumerate(group[1:], 1):
            pairs.append(dict(split='test', subset=SUBSET, group=start//3,
                              baseline_group=f'ordered_step_{offset}',
                              q_id=int(group[0]), k_id=int(key)))
    return pairs


def prior_camera_union(results_root, scene, excluded_roots):
    excluded, records = set(), []
    for path in sorted(Path(results_root).glob('*/'+scene+'/manifest.json')):
        if any(path.resolve().is_relative_to(root.resolve()) for root in excluded_roots):
            continue
        manifest = json.loads(path.read_text())
        if manifest.get('scene') != scene or 'frames' not in manifest:
            continue
        ids = {int(frame['image_id']) for frame in manifest['frames']}
        ids.update(int(pair[side+'_id']) for pair in manifest.get('pairs', []) for side in ('q', 'k'))
        excluded.update(ids)
        records.append(dict(path=str(path.resolve()), sha256=visual.sha256(path), camera_ids=sorted(ids)))
    if not records:
        raise ValueError('no prior scene manifests: refusing to declare cameras untouched without an exclusion audit')
    return excluded, records


def build_protocol(args):
    """Only metadata/available source geometry are inspected at selection time."""
    development_path = args.frozen_development/'summary.json'
    development = json.loads(development_path.read_text())
    projection_path = args.frozen_development/'layer2_projection.npy'
    if development['config']['layer'] != 'layer2' or visual.sha256(projection_path) != development['projections']['layer2']['sha256']:
        raise ValueError('the frozen development layer/projection no longer match')
    scenes = {}
    for scene in SCENES:
        source = Path(base.DEFAULT_SOURCES[scene])
        excluded, priors = prior_camera_union(args.results_root, scene, [args.out, args.visual_out])
        model, frames, ordered, geometry, dataset, normalization = base.load_scene(source)
        # load_scene's sort is by filename. No geometry/track quality enters.
        remaining = [int(image_id) for image_id in ordered if image_id not in excluded]
        pairs = ordered_pairs(remaining)
        used = sorted({pair[side+'_id'] for pair in pairs for side in ('q', 'k')})
        if set(used) & excluded:
            raise ValueError('fresh and previously exported camera IDs overlap')
        source_paths = [source/'gaussians.npz', source/'source_model.json', source/'training_scene.json',
                        source/'run_status.json', dataset]
        scenes[scene] = dict(source_output=str(source), dataset=str(dataset),
                             source_sha256={str(path):visual.sha256(path) for path in source_paths},
                             available_camera_ids=[int(image_id) for image_id in ordered],
                             excluded_camera_ids=sorted(excluded), prior_manifests=priors,
                             remaining_camera_ids_filename_order=remaining,
                             remaining_filenames=[frames[image_id]['name'] for image_id in remaining],
                             pairs=pairs, used_camera_ids=used,
                             unused_unpaired_camera_ids=sorted(set(remaining)-set(used)),
                             prior_used_intersection=[], selection_uses_track_or_matcher_metrics=False)
        del model, frames, geometry
    return dict(phase='fresh_camera_confirmation_data_preparation', subset=SUBSET,
                pair_rule='sort remaining images by filename; groups of up to 3; q=first, k=second and third; a last singleton is recorded unused',
                exclusion_rule='union of frames.image_id and all pair q_id/k_id in every existing scene data manifest',
                camera_id_meaning='posed image ID, not a shared intrinsic-camera calibration ID',
                source_scene_geometry_may_include_all_photos=True, matcher_evaluation_forbidden_in_this_export=True,
                frozen_visual=dict(development_summary=str(development_path.resolve()),
                    development_summary_sha256=visual.sha256(development_path),
                    projection=str(projection_path.resolve()),projection_sha256=visual.sha256(projection_path),
                    weights_sha256=development['model']['weights_sha256'],layer='layer2',
                    preprocessing='full RGB image BOX resize 192x144, ImageNet mean/std, no crop',
                    pooling='dense average over 6x8 grid',output_width=108,output_l2_normalized=True),
                source_sha256={str(Path(__file__).resolve()):visual.sha256(Path(__file__)),
                               str(Path(base.__file__).resolve()):visual.sha256(base.__file__),
                               str(Path(visual.__file__).resolve()):visual.sha256(visual.__file__)},
                scenes=scenes)


def validate_frozen_sources(protocol):
    frozen_visual = protocol['frozen_visual']
    if visual.sha256(frozen_visual['development_summary']) != frozen_visual['development_summary_sha256']:
        raise ValueError('development feature freeze changed')
    if visual.sha256(frozen_visual['projection']) != frozen_visual['projection_sha256']:
        raise ValueError('development projection freeze changed')
    for path, expected in protocol['source_sha256'].items():
        if visual.sha256(path) != expected:
            raise ValueError('controller/helper source changed after selection freeze')
    for scene in protocol['scenes'].values():
        for record in scene['prior_manifests']:
            if visual.sha256(record['path']) != record['sha256']:
                raise ValueError('prior exclusion manifest changed')
        for path, expected in scene['source_sha256'].items():
            if visual.sha256(path) != expected:
                raise ValueError('reconstruction source changed after selection freeze')


def export_raw(scene, selection, destination, protocol_hash):
    old_pair_function = base.make_pairs
    selected_pairs = selection['pairs']
    try:
        def frozen_pairs(_ordered_ids, _groups=8):
            return copy.deepcopy(selected_pairs)
        base.make_pairs = frozen_pairs
        manifest = base.export_scene(scene, Path(selection['source_output']), destination)
    finally:
        base.make_pairs = old_pair_function
    if base.make_pairs is not old_pair_function:
        raise ValueError('original pair function was not restored')
    actual = {int(frame['image_id']) for frame in manifest['frames']}
    if actual != set(selection['used_camera_ids']) or actual & set(selection['excluded_camera_ids']):
        raise ValueError('actual exported cameras violate selection freeze')
    manifest.update(subset=SUBSET, phase='fresh_camera_confirmation_data_preparation',
                    camera_split='previously unexported posed-image IDs, filename-ordered groups of up to three; all pairs test',
                    groups_per_split=None, fresh_selection=selection, selection_protocol_sha256=protocol_hash,
                    prior_camera_intersection=[], pair_selection_frozen_before_image_or_label_opening=True,
                    model_predictions_or_metrics_used=False)
    manifest['limitations'].append('Fresh cameras were unused by matcher/pair-route selection, but source photos may have trained the available GS/SfM reconstruction; this is conditional matching, not held-out-geometry reconstruction.')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def copy_frozen_visual(scene, raw_folder, destination, model, model_metadata, projection, development, protocol_hash):
    """Explicit confirmation phase; do not call/relax the old dev-only exporter."""
    manifest = json.loads((raw_folder/'manifest.json').read_text())
    if scene not in SCENES or manifest['subset'] != SUBSET or any(pair['split'] != 'test' for pair in manifest['pairs']):
        raise ValueError('not a frozen fresh-camera confirmation source')
    dataset_path = Path(manifest['dataset'])
    dataset = json.loads(dataset_path.read_text())
    used = set(manifest['fresh_selection']['used_camera_ids'])
    image_metadata = {image['id']:image for image in dataset['images'] if image['id'] in used}
    if image_metadata.keys() != used:
        raise ValueError('fresh image metadata missing')
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination/'image_cache'
    cache.mkdir(exist_ok=True)
    features, image_records = {}, []
    for image_id in sorted(used):
        image_path = (dataset_path.parent/image_metadata[image_id]['path']).resolve()
        image_hash = visual.sha256(image_path)
        if manifest['source_images_sha256'][str(image_path)] != image_hash:
            raise ValueError('fresh image changed between geometry and feature export')
        pooled, preparation = visual.extract_pooled(image_path, model)
        feature = visual.project_features(pooled['layer2_mean'], projection)
        if feature.shape != (48,108) or not np.isfinite(feature).all() or not np.allclose(np.linalg.norm(feature,axis=-1),1,atol=2e-6):
            raise ValueError('invalid frozen visual features')
        cache_path = cache/f'image_{image_id:06d}.npz'
        np.savez_compressed(cache_path, layer1_mean=pooled['layer1_mean'], layer2_mean=pooled['layer2_mean'], layer2_projected=feature)
        record = dict(image_id=image_id, image_path=str(image_path), image_sha256=image_hash,
                      preparation=preparation, cache_path=str(cache_path.resolve()), cache_sha256=visual.sha256(cache_path),
                      pooled_layer2_sha256=visual.array_sha256(pooled['layer2_mean']),
                      feature_sha256=visual.array_sha256(feature), weights_sha256=model_metadata['weights_sha256'],
                      projection_sha256=visual.array_sha256(projection.numpy()))
        cache_path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        features[image_id] = feature
        image_records.append(record)
    pair_records = []
    for record in manifest['pairs']:
        path = raw_folder/record['file']
        if visual.sha256(path) != record['npz_sha256']:
            raise ValueError('fresh raw pair changed')
        with np.load(path,allow_pickle=False) as data:
            arrays = {key:data[key].copy() for key in data.files}
        arrays['q_features'] = features[record['q_id']]
        arrays['k_features'] = features[record['k_id']]
        output = destination/record['file']
        np.savez_compressed(output,**arrays)
        pair = dict(record, source_npz_sha256=record['npz_sha256'], npz_sha256=visual.sha256(output),
                    nonfeature_array_sha256={key:visual.array_sha256(value) for key,value in arrays.items()
                                            if key not in ('q_features','k_features')})
        pair_records.append(pair)
    result = copy.deepcopy(manifest)
    result.update(pairs=pair_records, source_manifest=str((raw_folder/'manifest.json').resolve()),
                  source_manifest_sha256=visual.sha256(raw_folder/'manifest.json'),
                  feature_preparation=dict(phase='fresh_camera_confirmation_data_preparation',
                    permission='explicitly authorized new-phase extraction of previously unexported cameras; original development-only export_scene was not called or relaxed',
                    layer='layer2', pooling='dense regional average over 6x8 grid',
                    preprocessing='full RGB image BOX resize 192x144, ImageNet mean/std, no crop',
                    projection='exact frozen development layer2 128->108 QR projection',
                    projection_seed=visual.PROJECTION_SEED, output_l2_normalized=True,
                    frozen_development=development, model=model_metadata, image_features=image_records,
                    selection_protocol_sha256=protocol_hash, fitting_or_model_evaluation=False))
    (destination/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def verify_outputs(protocol, raw_root, frozen_root):
    required = {'q_features','k_features','q_mean','k_mean','q_cov','k_cov','q_valid','k_valid',
                'q_mean_alt','k_mean_alt','q_cov_alt','k_cov_alt','q_valid_alt','k_valid_alt',
                'q_pixel_mean','q_pixel_cov','q_projected_mean','q_projected_cov',
                'k_projected_mean','k_projected_cov','soft_labels','label_valid','track_counts',
                'K_query','R_query','t_query','K_key','R_key','t_key'}
    reports = []
    for scene, selection in protocol['scenes'].items():
        raw = json.loads((raw_root/scene/'manifest.json').read_text())
        frozen = json.loads((frozen_root/scene/'manifest.json').read_text())
        actual_ids = {int(frame['image_id']) for frame in raw['frames']}
        assert actual_ids == set(selection['used_camera_ids'])
        assert not actual_ids & set(selection['excluded_camera_ids'])
        assert len(raw['pairs']) == len(selection['pairs']) == len(frozen['pairs'])
        arrays_checked = 0
        for rp, fp in zip(raw['pairs'], frozen['pairs']):
            assert rp['split'] == fp['split'] == 'test'
            assert rp['subset'] == fp['subset'] == SUBSET
            assert visual.sha256(raw_root/scene/rp['file']) == rp['npz_sha256']
            assert visual.sha256(frozen_root/scene/fp['file']) == fp['npz_sha256']
            with np.load(raw_root/scene/rp['file'],allow_pickle=False) as a, np.load(frozen_root/scene/fp['file'],allow_pickle=False) as b:
                assert required <= set(a.files) == set(b.files)
                for key in a.files:
                    assert np.isfinite(a[key]).all(), (scene,rp['file'],key)
                    if key in ('q_features','k_features'):
                        assert a[key].shape == b[key].shape == (48,108)
                        assert np.isfinite(b[key]).all()
                        assert np.allclose(np.linalg.norm(b[key],axis=-1),1,atol=2e-6)
                    else:
                        assert np.array_equal(a[key],b[key],equal_nan=True),key
                        assert visual.array_sha256(b[key]) == fp['nonfeature_array_sha256'][key]
                        arrays_checked += 1
                target = a['track_counts']/np.maximum(a['track_counts'].sum(-1,keepdims=True),1)
                np.testing.assert_allclose(a['soft_labels'],target,atol=1e-7)
                assert np.array_equal(a['label_valid'],a['track_counts'].sum(-1)>=4)
                for side in ('q','k'):
                    for suffix in ('','_alt'):
                        assert a[side+'_mean'+suffix].shape == (48,3)
                        assert a[side+'_cov'+suffix].shape == (48,3,3)
                        assert a[side+'_valid'+suffix].shape == (48,)
        reports.append(dict(scene=scene, available_cameras=len(selection['available_camera_ids']),
                            excluded_cameras=len(selection['excluded_camera_ids']),
                            fresh_cameras=len(actual_ids), pair_count=len(raw['pairs']),
                            query_camera_count=len({pair['q_id'] for pair in raw['pairs']}),
                            nonfeature_arrays_verified=arrays_checked, prior_intersection=[],
                            raw_manifest_sha256=visual.sha256(raw_root/scene/'manifest.json'),
                            frozen_manifest_sha256=visual.sha256(frozen_root/scene/'manifest.json')))
    validate_frozen_sources(protocol)
    return dict(scenes=reports, schema_valid=True, labels_follow_unchanged_track_rule=True,
                source_hashes_unchanged=True, camera_sets_disjoint=True,
                frozen_nonfeature_arrays_identical=True, matcher_evaluation_run=False)


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    args.out.mkdir(parents=True,exist_ok=True)
    args.visual_out.mkdir(parents=True,exist_ok=True)
    protocol_path = args.out/'selection_protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        validate_frozen_sources(protocol)
    else:
        protocol = build_protocol(args)
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    protocol_hash = visual.sha256(protocol_path)
    snapshots = args.out/'source'
    snapshots.mkdir(exist_ok=True)
    for source_path in protocol['source_sha256']:
        shutil.copy2(source_path,snapshots/Path(source_path).name)
    for scene,selection in protocol['scenes'].items():
        export_raw(scene,selection,args.out/scene,protocol_hash)
    development_path = args.frozen_development/'summary.json'
    development_summary = json.loads(development_path.read_text())
    projection_path = args.frozen_development/'layer2_projection.npy'
    if visual.sha256(projection_path) != development_summary['projections']['layer2']['sha256']:
        raise ValueError('frozen development projection changed')
    projection = torch.from_numpy(np.load(projection_path,allow_pickle=False))
    if projection.shape != (128,108):
        raise ValueError('unexpected frozen projection width')
    model,model_metadata = visual.official_backbone(args.weights)
    if model_metadata['weights_sha256'] != development_summary['model']['weights_sha256']:
        raise ValueError('frozen backbone weights changed')
    if development_summary['config']['layer'] != 'layer2':
        raise ValueError('development active layer is not the requested frozen layer2')
    development = dict(summary=str(development_path.resolve()),summary_sha256=visual.sha256(development_path),
                       projection=str(projection_path.resolve()),projection_sha256=visual.sha256(projection_path),
                       preprocessing_unchanged=True, active_layer_unchanged=True)
    shutil.copy2(projection_path,args.visual_out/'layer2_projection.npy')
    for scene in protocol['scenes']:
        copy_frozen_visual(scene,args.out/scene,args.visual_out/scene,model,model_metadata,projection,development,protocol_hash)
    verification = verify_outputs(protocol,args.out,args.visual_out)
    verification['selection_protocol_sha256'] = protocol_hash
    verification['elapsed_seconds'] = time.monotonic()-started
    for root in (args.out,args.visual_out):
        (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(dict(protocol_sha256=protocol_hash,verification=verification),indent=2)+'\n')
    (args.visual_out/'summary.json').write_text(json.dumps(dict(protocol_sha256=protocol_hash,model=model_metadata,
                        frozen_development=development,verification=verification),indent=2)+'\n')
    readme = '''# Previously unexported-camera data: confirmation preparation only

Pair selection was written to selection_protocol.json before new image pixels or pair labels were opened. Every prior scene-data manifest's frames/q_id/k_id union is excluded. Camera here means posed-image ID, not an intrinsic calibration ID. Remaining images are sorted by filename and grouped in threes: the first queries the second and third; a final pair is retained, a singleton is recorded unused. No shared-track threshold, prediction or matching score selects pairs. All selected pairs remain even if their labels/geometry have no valid rows.

The original geometry exporter is wrapped with a frozen make_pairs callback and restored in finally; its source file is unchanged. Source snapshots/hashes cover the actual controller, exporter and visual helper. Raw RGB108, independent even/odd-GS moment fits and the alternate swap, native query-region covariance, projected q/k support, cameras and sparse SfM-track targets retain the original schema. All pairs have split=test and subset=previously_unexported_cameras.

The sibling round3_fresh_frozen_visual_data is an explicitly authorized confirmation-preparation phase using the same official local ResNet18 weights, layer2 regional average pooling, exact saved development 128->108 projection, ImageNet normalization and full 192x144 BOX-resize. It calls the frozen extraction primitives, not a bypass of the old development-only exporter. Nonfeature arrays are verified identical. No layer, projection or preprocessing was fitted or selected here.

No matcher has been trained or evaluated on this export. The main thread must freeze candidate/settings before evaluation. Tea/kitchen have very few remaining cameras; the data imbalance is retained. New cameras are from already known scenes and can neighbor previously exported frames, so this is conditional matching on unused matcher images, not independent new-scene evidence. Source photographs may already have trained the available GS/SfM reconstruction. Sparse-track targets are not dense coverage or occlusion truth.

verification.json proves the source hashes, fresh/prior ID disjointness, output schema and exact raw/frozen nonfeature equality. It contains no model performance metrics.

Reproduce with `python -m experiments.fresh_real_pairs`.
'''
    (args.out/'README.md').write_text(readme)
    (args.visual_out/'README.md').write_text(readme+'\nThe selection protocol and source snapshots are in the sibling raw-data directory.\n')
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path('results'))
    parser.add_argument('--out',type=Path,default=Path('results/round3_fresh_real_data'))
    parser.add_argument('--visual-out',type=Path,default=Path('results/round3_fresh_frozen_visual_data'))
    parser.add_argument('--frozen-development',type=Path,default=Path('results/round3_frozen_visual_data'))
    parser.add_argument('--weights',type=Path,default=Path('/home/liwt/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    main(parser.parse_args())
