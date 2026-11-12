"""Source-only visual feature binding and an exactly byte-bounded raw-token control."""
import argparse
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.gaussian_scene_data import OUT, CACHE, SOURCES, input_plan, source_path, read
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_memory_coarse_pe import patch_tokens
from experiments.gaussian_memory_incremental_data import save_tensor, write_json
from experiments.gaussian_memory_setup import sha256
from models.gaussian_scene_memory import field_bytes, field_to_device, lift_source_features


@torch.no_grad()
def fps(means, count):
    distance = torch.full((len(means),), float('inf'), device=means.device)
    indices = torch.empty(count, dtype=torch.long, device=means.device)
    index = 0
    for j in range(count):
        indices[j] = index
        torch.minimum(distance, (means - means[index]).square().sum(-1), out=distance)
        distance[indices[:j + 1]] = -1
        index = int(distance.argmax())
    return indices


def load_tensor(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def token_control(field,source,raw,registration,features):
    """Depth at feature-cell centers, using nearest-exact rather than legacy top-left sampling."""
    K=source['K'].clone();K[:,:2]/=14
    depth=F.interpolate(raw['source_depth'].permute(0,3,1,2),(32,32),mode='nearest-exact')[:,0]*registration['scale']
    yy,xx=torch.meshgrid(torch.arange(32,device='cuda')+.5,torch.arange(32,device='cuda')+.5,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    camera=torch.einsum('ni,vji->vnj',pixels,torch.linalg.inv(K))*depth.reshape(4,-1,1)
    means=torch.einsum('vni,vji->vnj',camera,source['c2w'][:,:3,:3])+source['c2w'][:,None,:3,3]
    common={key:field[key] for key in ('unit','source_c2w','source_intrinsic','source_ids')}
    count=min(4096,(field_bytes(field)-field_bytes(common))//(1024*2+3*4+1))
    indices=fps(means.reshape(-1,3),count)
    return dict(common,means=means.reshape(-1,3)[indices].float(),features=features.reshape(-1,1024)[indices].half(),
                has_content=torch.ones(count,device='cuda',dtype=torch.bool))


@torch.no_grad()
def ensure_token_centers(model,p):
    """Versioned pre-T-fit correction; leaves every Gaussian field and L input unchanged."""
    revised={}
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];marker=OUT/'binding'/f'{rid}.json';record=read(marker)
        if record.get('token_geometry_version',1)==2:continue
        folder=CACHE/rid;backup=folder/'token_memory_top_left_v1.pt'
        archive=OUT/'failures/token_geometry_v1'/f'{rid}.json';archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():write_json(archive,record)
        if not backup.exists():shutil.copyfile(folder/'token_memory.pt',backup)
        assert sha256(backup)==read(archive)['files'][str(folder/'token_memory.pt')]
        field=load_tensor(folder/'scene_memory.pt','cuda');original_field_sha=sha256(folder/'scene_memory.pt')
        source=load_tensor(source_path(rid),'cuda');raw=load_tensor(folder/'anysplat_source_init.pt','cuda')
        registration=load_tensor(folder/'source_registration.pt','cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=torch.cat([patch_tokens(model,image) for image in source['rgb'].float()/255]).reshape(4,32,32,1024)
        tokens=token_control(field,source,raw,registration,features)
        save_tensor(folder/'token_memory.pt',tokens)
        assert sha256(folder/'scene_memory.pt')==original_field_sha
        record['files'][str(folder/'token_memory.pt')]=sha256(folder/'token_memory.pt')
        record.update(token_geometry_version=2,token_depth_sampling='nearest-exact to32x32 cell centers; pixel7+14n',
                      serialized_file_bytes={name:(folder/name).stat().st_size for name in ('scene_memory.pt','token_memory.pt')},
                      token_center_repair='Pre-T-fit correction, old token tensor and binding manifest preserved; Gaussian field unchanged')
        write_json(marker,record);revised[rid]=sha256(marker)
        del field,source,raw,registration,features,tokens
    if revised:
        write_json(OUT/'failures/token_center_sampling_repair_v1.json',dict(complete=True,
            cause='Legacy nearest448to32 selected depth at0+14n while feature rays are at cell centers; nearest-exact selects7+14n',
            correction='Center-sampled source predicted depth for T; same byte budget and FPS, no query data used',
            before_first_T_formal_fit=True,Gaussian_fields_unchanged=True,L_training_unchanged=True,
            development_scores_not_used=True,updated_binding_sha256=revised))


@torch.no_grad()
def bind_one(model, meta):
    rid = meta['id']; folder = CACHE / rid
    marker = OUT / 'binding' / f'{rid}.json'
    if marker.exists():
        record = read(marker)
        assert all(sha256(path) == digest for path, digest in record['files'].items())
        return
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    source = load_tensor(source_path(rid), 'cuda')
    field = load_tensor(folder / 'coarse_geometry.pt', 'cuda')
    raw = load_tensor(folder / 'anysplat_source_init.pt', 'cuda')
    registration = load_tensor(folder / 'source_registration.pt', 'cuda')
    images = source['rgb'].float() / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = torch.cat([patch_tokens(model, image) for image in images]).reshape(4, 32, 32, 1024)
    K = source['K'].clone(); K[:, :2] /= 14
    field['features'], field['mass'], field['has_content'] = lift_source_features(field, features, source['c2w'], K)
    field['source_ids'] = torch.tensor(SOURCES, dtype=torch.int32, device='cuda')
    budget = field_bytes(field)
    # Raw-token centers use the same source-only predicted depth scale and the given source cameras.
    # No query depth, query RGB or supervision is accessible to this function.
    tokens = token_control(field,source,raw,registration,features)
    count = len(tokens['means'])
    assert field_bytes(tokens) <= budget
    files = {}
    for name, value in [('scene_memory.pt', field), ('token_memory.pt', tokens)]:
        path = folder / name
        save_tensor(path, field_to_device(value, 'cpu'))
        reloaded = load_tensor(path, 'cuda')
        assert set(value) == set(reloaded)
        assert all(torch.equal(tensor, reloaded[key]) for key, tensor in value.items())
        files[str(path)] = sha256(path)
    torch.cuda.synchronize()
    record = dict(complete=True, id=rid, source_only=True, source_input_sha256=sha256(source_path(rid)),
                  coarse_geometry_sha256=sha256(folder/'coarse_geometry.pt'), files=files,
                  gaussian_count=len(field['means']), content_valid=int(field['has_content'].sum()),
                  channels=1024, feature_dtype='float16', scene_bytes=budget, token_count=count,
                  token_bytes=field_bytes(tokens), tensor_save_reload_exact=True,
                  token_geometry_version=2,token_depth_sampling='nearest-exact to32x32 cell centers; pixel7+14n',
                  serialized_file_bytes={name:(folder/name).stat().st_size for name in ('scene_memory.pt','token_memory.pt')},
                  feature_grid=[32,32], source_feature_model_sha256=input_plan()['model_sha256'],
                  token_center_convention='source predicted depth times common registration scale; unproject given source K/c2w',
                  elapsed_seconds=time.monotonic()-started, peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
    write_json(marker, record); print(record, flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true'); args=parser.parse_args()
    plan=input_plan(); rows=plan['training_catalog'][:2] if args.pilot else plan['training_catalog']+plan['catalog']
    model=model_load(plan)
    for meta in rows: bind_one(model, meta)


if __name__ == '__main__': main()
