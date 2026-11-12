"""Official AnySplat initialization in an isolated source-only process."""
import argparse
import importlib.metadata
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_scene_setup import SOURCE, OUT, write
from experiments.gaussian_scene_data import CACHE, input_plan, source_path, read
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_incremental_data import save_tensor


def model_load():
    sys.path[:0] = [str(ROOT/'third_party/gaussian_scene_python'),
                   str(ROOT/'third_party/native_history_python'), str(SOURCE)]
    # Skip training dataset/model registries. Concrete inference implementations
    # are still imported from their pinned files, not replaced with stubs.
    for name in ('src', 'src.dataset', 'src.model', 'src.model.encoder', 'src.model.model'):
        package = types.ModuleType(name)
        package.__path__ = [str(SOURCE.joinpath(*name.split('.')))]
        sys.modules[name] = package
    from src.model.model.anysplat import AnySplat
    from src.model.encoder.anysplat import EncoderAnySplatCfg
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
    from src.model.encoder.vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    identity = read(OUT/'setup/anysplat_weights.json')
    folder = Path(identity['folder'])
    assert sha256(folder/'model.safetensors') == identity['files']['model.safetensors']
    config = read(folder/'config.json')
    own = VGGT.__dict__.get('from_pretrained')
    # The AnySplat checkpoint supplies ALL retained backbone/head parameters.
    # Avoid an unnecessary second VGGT download and initialization copy; require
    # strict full-state loading below, so no random parameter can go unnoticed.
    VGGT.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls())
    try:
        model = AnySplat(EncoderAnySplatCfg(**config['encoder_cfg']),
                         DecoderSplattingCUDACfg(**config['decoder_cfg']))
    finally:
        if own is None:
            del VGGT.from_pretrained
        else:
            VGGT.from_pretrained = own
    state = load_file(folder/'model.safetensors')
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    count = len(state)
    del state
    model = model.eval().requires_grad_(False).cuda()
    write(OUT/'setup/anysplat_runtime.json', dict(
        strict_full_state_keys=count, checkpoint_sha256=identity['files']['model.safetensors'],
        constructor_backbone_download_skipped=True, all_retained_weights_loaded=True,
        training_registries_skipped=True, optional_import_patch='optional_import_compatibility.json',
        actual_encoder_sha256=sha256(SOURCE/'src/model/encoder/anysplat.py'),
        actual_head_modules_sha256=sha256(SOURCE/'src/model/encoder/heads/head_modules.py'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in
                  ['e3nn','torch-scatter','omegaconf','hydra-core']}))
    return model


@torch.inference_mode()
def initialize(model, meta):
    rid = meta['id']
    marker = OUT/'source_initialization'/f'{rid}.json'
    path = CACHE/rid/'anysplat_source_init.pt'
    if marker.exists():
        previous = read(marker)
        assert previous['complete'] and previous['output_sha256'] == sha256(path)
        assert previous['source_payload_sha256'] == sha256(source_path(rid))
        return
    payload = torch.load(source_path(rid), map_location='cpu', weights_only=False)
    assert set(payload) == {'rgb','c2w','K','source_ids','id','place','split'}
    assert payload['source_ids'] == [0,3,6,9]
    images = payload['rgb'][None].cuda().float()/255.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    output = model.encoder(images, global_step=0, visualization_dump=None)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    gaussians = output.gaussians
    result = {name:getattr(gaussians, name)[0].float().cpu().contiguous()
              for name in ('means','covariances','harmonics','opacities','scales','rotations')}
    result.update(pred_c2w=output.pred_context_pose['extrinsic'][0].float().cpu(),
                  pred_intrinsic_normalized=output.pred_context_pose['intrinsic'][0].float().cpu(),
                  source_depth=output.depth_dict['depth'][0].float().cpu(),
                  source_confidence_mask=output.depth_dict['conf_valid_mask'][0].cpu())
    finite = {k:bool(torch.isfinite(v).all()) for k,v in result.items() if v.is_floating_point()}
    assert all(finite.values()), finite
    assert (result['scales']>0).all() and (result['source_depth']>0).all()
    save_tensor(path, result)
    write(marker, dict(complete=True, id=rid, source_ids=payload['source_ids'],
                       source_payload_sha256=sha256(source_path(rid)),
                       checkpoint_sha256=read(OUT/'setup/anysplat_weights.json')['files']['model.safetensors'],
                       output_path=str(path), output_sha256=sha256(path),
                       gaussians=len(result['means']), source_inference_seconds=seconds,
                       peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       finite=finite, queries_used=False,
                       camera_convention='Official output extrinsic is already camera-to-world; K is normalized by image width/height.'))
    print({'source_initialized':rid, 'gaussians':len(result['means']), 'seconds':seconds}, flush=True)
    del output,gaussians,result,images
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    p = input_plan()
    catalog = p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    model = model_load()
    for meta in catalog:
        initialize(model,meta)


if __name__ == '__main__':
    main()
