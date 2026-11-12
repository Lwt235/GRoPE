"""Separate source-only payloads from supervised query payloads for the scene task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import sensor_target, depth_source, save_tensor, write_json

OUT = ROOT / 'results/gaussian_memory/gaussian_scene_memory1'
CACHE = DATA / 'gaussian_scene_memory_v1'
SOURCES = (0, 3, 6, 9)
QUERIES = (2, 5, 8, 11)
CONDITIONS = ('full', 'missing')
KINDS = ('L', 'T', 'C', 'P', 'G')
SEEDS = (20261110, 20261111, 20261112)


def read(path):
    return json.loads(Path(path).read_text())


def source_path(rid):
    return CACHE / rid / 'source_input.pt'


def query_path(rid, frame):
    return CACHE / rid / f'query_{frame}.pt'


def transformed_rgb(path):
    """Match the pinned AnySplat process_image resize/crop, retaining byte RGB."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        new_height = 448 if width > height else int(height * (448 / width))
        new_width = int(width * (448 / height)) if width > height else 448
        left, top = (new_width - 448) // 2, (new_height - 448) // 2
        resized = image.resize((new_width, new_height))
        result = resized.crop((left, top, left + 448, top + 448))
        pixels = torch.from_numpy(np.asarray(result).copy()).permute(2, 0, 1)
    return pixels, (new_width, new_height, left, top)


def transformed_intrinsics(K, transform, size=448):
    # Existing calibration is defined on the 280x210 sensor scoring grid.
    new_width, new_height, left, top = transform
    K = torch.as_tensor(K, dtype=torch.float32).clone()
    K[0] *= new_width / 280
    K[1] *= new_height / 210
    K[0, 2] -= left
    K[1, 2] -= top
    K[:2] *= size / 448
    return K


def transformed_target(frame, transform):
    depth, valid = sensor_target(frame)
    new_width, new_height, left, top = transform
    def warp(tensor):
        image = Image.fromarray(tensor.numpy())
        image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        image = image.crop((left, top, left + 448, top + 448))
        return torch.from_numpy(np.asarray(image.resize((210, 210), Image.Resampling.NEAREST)).copy())
    return warp(depth).float(), warp(valid).bool()


def query_image(payload, condition):
    image = payload['rgb'].float() / 255.
    if condition == 'missing':
        fill = image.new_tensor([.485, .456, .406])[:, None, None]
        image = torch.where(payload['image_missing'][None], fill, image)
    elif condition != 'full':
        raise ValueError(condition)
    return image


def input_plan():
    path = OUT / 'input_plan.json'
    if path.exists():
        return read(path)
    parent_path = ROOT / 'results/gaussian_memory/native_gaussian_pe1/plan.json'
    parent = read(parent_path)
    train, development = parent['training_catalog'], parent['catalog']
    assert len(train) == len(development) == 12
    assert not {m['place'] for m in train} & {m['place'] for m in development}
    assert all(len(m['row']['frames']) >= 12 for m in train + development)
    result = dict(experiment='gaussian_scene_memory1', training_catalog=train,
                  catalog=development, sources=list(SOURCES), queries=list(QUERIES),
                  conditions=list(CONDITIONS), kinds=list(KINDS), seeds=list(SEEDS),
                  original_frame_counts={m['id']:len(m['row']['frames']) for m in train + development},
                  selected_window='first twelve frames; only fixed source/query indices used',
                  preprocessing='Pinned AnySplat process_image: resize shorter side to448, center crop448; same transforms for all methods and given intrinsics',
                  scoring_hw=[210, 210], missing_rectangle_image=[112, 336, 112, 336],
                  all_data_previously_used=True, new_confirmation=False,
                  model_path=parent['model_path'], model_sha256=parent['model_sha256'],
                  github_commit=parent['github_commit'], hf_revision=parent['hf_revision'],
                  official_source_sha256=parent['official_source_sha256'],
                  parent_plan_sha256=sha256(parent_path), expected_adapted_predictions=1440)
    write_json(path, result)
    return result


def prepare_one(meta, split):
    rid, row = meta['id'], meta['row']
    marker = OUT / 'inputs' / f'{rid}.json'
    if marker.exists():
        record = read(marker)
        for path, digest in record['files'].items():
            assert sha256(path) == digest
        return
    folder = CACHE / rid
    folder.mkdir(parents=True, exist_ok=True)
    raw_files, files = {}, {}
    images, intrinsics, poses = [], [], []
    # No query image, query depth or query teacher is loaded in this loop.
    for index in SOURCES:
        frame = row['frames'][index]
        assert sha256(frame['rgb_path']) == frame['rgb_sha256']
        rgb, transform = transformed_rgb(frame['rgb_path'])
        images.append(rgb)
        intrinsics.append(transformed_intrinsics(row['K'][index], transform))
        poses.append(torch.tensor(row['c2w_metric'][index], dtype=torch.float32))
        raw_files[frame['rgb_path']] = frame['rgb_sha256']
    source = dict(rgb=torch.stack(images), c2w=torch.stack(poses), K=torch.stack(intrinsics),
                  source_ids=list(SOURCES), id=rid, place=meta['place'], split=split)
    assert set(source) == {'rgb', 'c2w', 'K', 'source_ids', 'id', 'place', 'split'}
    save_tensor(source_path(rid), source)
    files[str(source_path(rid))] = sha256(source_path(rid))
    image_missing = torch.zeros(448, 448, dtype=torch.bool)
    image_missing[112:336, 112:336] = True
    hidden = torch.from_numpy(np.asarray(Image.fromarray(image_missing.numpy()).resize((210, 210), Image.Resampling.NEAREST)).copy())
    for index in QUERIES:
        frame = row['frames'][index]
        assert sha256(frame['rgb_path']) == frame['rgb_sha256']
        raw_files[frame['rgb_path']] = frame['rgb_sha256']
        depth_file = depth_source(frame)
        digest = sha256(depth_file)
        if 'depth_sha256' in frame:
            assert digest == frame['depth_sha256']
        raw_files[str(depth_file)] = digest
        rgb, transform = transformed_rgb(frame['rgb_path'])
        target, valid = transformed_target(dict(frame, depth_path=str(depth_file)), transform)
        value = dict(rgb=rgb, c2w=torch.tensor(row['c2w_metric'][index], dtype=torch.float32),
                     K=transformed_intrinsics(row['K'][index], transform),
                     K_metric=transformed_intrinsics(row['K'][index], transform, 210),
                     target=target, valid=valid, hidden=hidden, image_missing=image_missing,
                     id=rid, place=meta['place'], query=index, split=split)
        path = query_path(rid, index)
        save_tensor(path, value)
        files[str(path)] = sha256(path)
    write_json(marker, dict(complete=True, id=rid, split=split, files=files, raw_files=raw_files,
                           source_payload_no_query_inputs=True, source_ids=list(SOURCES),
                           queries=list(QUERIES), plan_sha256=sha256(OUT/'input_plan.json')))
    print(json.dumps({'input_ready':rid, 'split':split}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    p = input_plan()
    rows = [(m, 'training') for m in p['training_catalog']]
    if args.pilot:
        rows = rows[:2]
    else:
        rows += [(m, 'development') for m in p['catalog']]
    for meta, split in rows:
        prepare_one(meta, split)


if __name__ == '__main__':
    main()
