"""Prospective cohort, immutable old models, and reference/query-separated caches."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_transport_data import OUT as OLD, checked as old_checked
from experiments.gaussian_memory_surface_data import load_record as old_load, coarse_depth
from models.gaussian_memory_transport import GeometryTransport, frozen_memory
from models.gaussian_memory_global_summary import GlobalSummary
from models.gaussian_memory_surface import SurfaceMemory

OUT = ROOT / 'results/gaussian_memory/global_summary1'
CACHE = DATA / 'global_summary_v1'
SOURCE = DATA.parent / 'region_rotary_confirmation_20261110/raw'
SEEDS = (20261110, 20261111, 20261112)
KINDS = ('gaussian', 'center', 'global', 'local', 'large_local')
REF = [0, 4, 8, 12]


def path(kind, seed, initial=False):
    filename = 'initial.pt' if initial else 'step_2000.pt'
    if kind == 'global': return OUT / f'global_{seed}' / filename
    if kind == 'large_local': return ROOT / f'results/gaussian_memory/surface1/local_{seed}' / filename
    dirname = f'none_center_local_{seed}' if kind == 'local' else f'predicted_{kind}_latent_{seed}'
    return OLD / dirname / filename


def model(kind, plan):
    if kind == 'global': return GlobalSummary(plan['center_variance'])
    if kind == 'large_local': return SurfaceMemory('local', 191, plan['center_variance'])
    return GeometryTransport('local' if kind == 'local' else 'latent',
                             'center' if kind == 'local' else kind, plan['center_variance'], 67)


def prepare():
    old = old_checked(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    source = SOURCE / 'selection.json'
    selection = json.loads(source.read_text())['scenes']
    used = {m['row']['scene'].split('/')[0] for m in old['catalog']}
    assert len(selection) == 20 and not used & {r['scene'].split('/')[0] for r in selection}
    records = []; inputs = {str(source): sha256(source), str(OLD/'plan.json'): sha256(OLD/'plan.json'),
                            str(DATA/'cache_v1/pca.pt'): sha256(DATA/'cache_v1/pca.pt'),
                            str(DATA/'source_identity.json'): sha256(DATA/'source_identity.json')}
    for source_row in sorted(selection, key=lambda r:r['scene']):
        scene = source_row['scene']; frames = sorted(source_row['frames'], key=lambda f:f['index'])[:13]
        assert len(frames) == 13
        row = dict(id=f'confirm_{len(records):02d}', scene=scene, split='confirmation',
                   root=str(SOURCE), intrinsics=str(SOURCE/scene/'intrinsics.txt'),
                   poses=str(SOURCE/source_row['pose_file']), reference=REF, training_query=[],
                   heldout_query=[1,2,3,5,6,7,9,10,11], frames=[])
        for key in ('intrinsics','poses'):
            row[key+'_sha256'] = sha256(row[key]); inputs[row[key]] = row[key+'_sha256']
        assert row['intrinsics_sha256'] == source_row['K_sha256']
        assert row['poses_sha256'] == source_row['pose_sha256']
        for frame in frames:
            depth = SOURCE/scene/'depth'/frame['depth']; assert depth.is_file()
            entry = dict(index=frame['index'], rgb_path=str(CACHE/'rgb'/scene/frame['rgb']),
                         rgb_url=source_row['base_url']+'image/'+frame['rgb'],
                         depth_path=str(depth), depth_sha256=sha256(depth), time_gap_us=frame['time_gap_us'])
            inputs[str(depth)] = entry['depth_sha256']; row['frames'].append(entry)
        records.append(row)
    old_models = {str(path(k,s)):sha256(path(k,s)) for k in KINDS if k!='global' for s in SEEDS}
    sources = dict(old['source_sha256'])
    names = list(ROOT.glob('experiments/gaussian_memory_summary_*.py')) + [ROOT/'models/gaussian_memory_global_summary.py']
    names += [ROOT/'experiments'/n for n in ('gaussian_memory_data.py','gaussian_memory_view_data.py','gaussian_memory_surface_data.py','gaussian_memory_surface_evaluate.py','gaussian_memory_surface_diagnostic.py','gaussian_memory_coverage_report.py','gaussian_memory_surface_train.py')]
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in names})
    plan = {**{k:old[k] for k in ('catalog','training_ids','primary_places','steps','batch','lr','weight_decay','center_variance','gradient_clip')},
            'seeds':SEEDS, 'kinds':KINDS, 'cohort':records, 'old_places':sorted(used),
            'new_places':20, 'method_unseen_confirmation':True, 'historically_untouched_claim':False,
            'history':'These depth images were used in earlier rotary-region registration; none of these places entered the shared-memory route.',
            'input_sha256':inputs,'old_models_sha256':old_models,'source_sha256':sources,
            'document_sha256':sha256(ROOT/'docs/gaussian_memory_global_summary_plan.md')}
    for seed in SEEDS:
        initial = torch.load(path('gaussian',seed,True), map_location='cpu', weights_only=False)
        net = model('global',plan); net.load_state_dict(initial['model'])
        assert sum(p.numel() for p in net.parameters()) == 145477
        directory = path('global',seed).parent; directory.mkdir()
        torch.save({'model':net.state_dict(),'config':{'kind':'global','seed':seed,'parameters':145477,'steps':2000}},path('global',seed,True))
        inputs[str(path('global',seed,True))] = sha256(path('global',seed,True))
    write_json(OUT/'plan.json',plan)
    print(json.dumps({'prepared':True,'new_fits':3,'fixed_old_models':12,'method_unseen_places':20,'RGB_files':260}),flush=True)


def checked():
    plan = json.loads((OUT/'plan.json').read_text())
    assert sha256(ROOT/'docs/gaussian_memory_global_summary_plan.md') == plan['document_sha256']
    for name,digest in plan['source_sha256'].items(): assert sha256(ROOT/name)==digest,name
    for name,digest in plan['old_models_sha256'].items(): assert sha256(name)==digest,name
    return plan


def locked():
    plan=checked(); lock=json.loads((OUT/'evaluation_lock.json').read_text())
    assert sha256(OUT/'plan.json')==lock['plan_sha256']
    for name,digest in lock['models'].items(): assert sha256(name)==digest,name
    return plan


def download():
    import requests
    plan=checked(); CACHE.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'download_complete.json').exists()
    session=requests.Session();session.trust_env=False
    rows=[];failures=[];start=time.monotonic()
    for row in plan['cohort']:
        for frame in row['frames']:
            dest=Path(frame['rgb_path']);dest.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(5):
                try:
                    if not dest.exists():
                        assert frame['rgb_url'].startswith('https://sun3d.cs.princeton.edu/data/')
                        response=session.get(frame['rgb_url'],timeout=(20,45));response.raise_for_status()
                        partial=dest.with_suffix('.jpg.partial');partial.write_bytes(response.content)
                        with Image.open(partial) as im: im.verify()
                        partial.replace(dest)
                    with Image.open(dest) as im: im.verify()
                    rows.append({'path':str(dest),'sha256':sha256(dest),'bytes':dest.stat().st_size});break
                except Exception as error:
                    failures.append({'path':str(dest),'attempt':attempt+1,'type':type(error).__name__})
                    write_json(OUT/'download_failures.json',{'failures':failures})
                    if attempt==4: raise
                    time.sleep(min(5*(attempt+1),20))
        write_json(OUT/'download_progress.json',{'downloaded':len(rows),'total':260,'proxy_used':False})
        print(json.dumps({'RGB_downloaded':len(rows),'total':260}),flush=True)
    write_json(OUT/'download_complete.json',{'rows':rows,'bytes':sum(r['bytes'] for r in rows),'seconds':time.monotonic()-start,'failures':failures,'proxy_used':False})


@torch.inference_mode()
def extract():
    from experiments.gaussian_memory_data import load_images,source_geometry,merge_seeds
    from experiments.gaussian_memory_view_data import load_backbone,cache_tokens
    from experiments.round5_rgbd_data import read_sun3d_intrinsics,read_sun3d_extrinsics,resize_intrinsics,_listing
    plan=locked();torch.set_num_threads(4);torch.manual_seed(20261110)
    hashes=json.loads((OUT/'download_complete.json').read_text())['rows'];rgb_hash={r['path']:r['sha256'] for r in hashes}
    backbone=load_backbone();pca=torch.load(DATA/'cache_v1/pca.pt',map_location='cuda',weights_only=False)
    catalog=[];identities={};diagnostics=[];start=time.monotonic()
    for original in plan['cohort']:
        row={**original,'frames':[{**f,'rgb_sha256':rgb_hash[f['rgb_path']]} for f in original['frames']]}
        directory=CACHE/row['id'];directory.mkdir(exist_ok=True)
        # Completed items may be resumed only if their prior identities still match.
        item_path=directory/'identity.json'
        if item_path.exists():
            item=json.loads(item_path.read_text())
            for p,digest in item['sha256'].items():assert sha256(p)==digest,p
            catalog.append(item['meta']);identities.update(item['sha256']);diagnostics.append(item['diagnostics']);continue
        for key in ('intrinsics','poses'):assert sha256(row[key])==row[key+'_sha256']
        names=_listing(SOURCE/row['scene']/'image_index.html','jpg')
        for f in row['frames']:assert names[f['index']]==Path(f['rgb_path']).name
        poses=read_sun3d_extrinsics(row['poses'])[[f['index'] for f in row['frames']]]
        poses=np.linalg.inv(poses[0])[None]@poses
        intrinsic=read_sun3d_intrinsics(row['intrinsics']);Ks=[]
        for f in row['frames']:
            with Image.open(f['rgb_path']) as im:k=resize_intrinsics(intrinsic,(im.height,im.width),(210,280))
            k[:2,2]+=.5;Ks.append(k)
        row.update(K=np.stack(Ks).tolist(),c2w_metric=poses.tolist())
        images=load_images(row,'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            refs=images[REF][None];tokens,patch_start=backbone.aggregator(refs)
            normalized=(images[REF]-backbone.aggregator._resnet_mean[0])/backbone.aggregator._resnet_std[0]
            teacher=backbone.aggregator.patch_embed(normalized)['x_norm_patchtokens'].float()
            with torch.autocast('cuda',enabled=False):
                pose=backbone.camera_head(tokens)[-1]
                depth,conf=backbone.depth_head(tokens,images=refs,patch_start_idx=patch_start)
                K=torch.tensor(row['K'],device='cuda').float();camera=torch.tensor(row['c2w_metric'],device='cuda').float()
                means,cov,valid,camera,diag=source_geometry(depth,conf,pose,K,camera)
                mean,chol,near,weights=merge_seeds(means,cov,valid)
                content=(teacher.reshape(-1,1024)[near]*weights[...,None]).sum(1)
                raw={'seed_mean':mean.cpu(),'seed_chol':chol.cpu(),
                     'seed_teacher':((content-pca['mean'])@pca['basis']/pca['scale']).cpu(),
                     'K':(K/K.new_tensor([14,14,1])[:,None]).cpu(),'c2w':camera.cpu(),
                     'diagnostics':diag,'episode':row,'rgb':(images.permute(0,2,3,1)*255).round().byte().cpu()}
        del tokens,teacher,refs,images,depth,conf,pose
        torch.save(raw,directory/'raw.pt')
        memory=frozen_memory(raw)
        torch.save({'memory':memory,'unit':diag['unit_meters'],'reference_frames':REF,
                    'query_RGB_used':False,'query_depth_used':False,'sensor_used':False},directory/'memory.pt')
        cache_tokens(backbone,raw,directory/'tokens.pt')
        target_depth=torch.ones(13,300);target_valid=torch.zeros(13,300,dtype=torch.bool);fractions=torch.zeros(13,300)
        sources=[]
        for q in row['heldout_query']:
            f=row['frames'][q];assert sha256(f['depth_path'])==f['depth_sha256']
            d,v,frac=coarse_depth(f['depth_path']);assert v.any(),(row['id'],q)
            target_depth[q]=torch.from_numpy(d);target_valid[q]=torch.from_numpy(v);fractions[q]=torch.from_numpy(frac)
            sources.append({'query':q,'path':f['depth_path'],'sha256':f['depth_sha256']})
        torch.save({'depth_m':target_depth,'log_depth_m':target_depth.log(),'valid':target_valid,
                    'fractions':fractions,'sources':sources,'supervision_only':True},directory/'target.pt')
        meta={'id':row['id'],'row':row,'eval_split':'confirmation',
              **{k:str(directory/(filename+'.pt')) for k,filename in [('raw','raw'),('tokens','tokens'),('memory','memory'),('target','target')]}}
        hashes={meta[k]:sha256(meta[k]) for k in ('raw','tokens','memory','target')}
        item={'meta':meta,'sha256':hashes,'diagnostics':{'id':row['id'],**diag,'valid_label_fraction':float(target_valid[row['heldout_query']].float().mean())}}
        write_json(item_path,item);catalog.append(meta);identities.update(hashes);diagnostics.append(item['diagnostics'])
        print(json.dumps({'extracted':row['id'],'completed':len(catalog),'places':20,'scale_ratio':diag['reference_scale_ratio_max_over_min']}),flush=True)
    write_json(OUT/'cache_identity.json',{'catalog':catalog,'sha256':identities,'diagnostics':diagnostics,
               'seconds':time.monotonic()-start,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,
               'evaluation_lock_sha256':sha256(OUT/'evaluation_lock.json'),'PCA_refit':False})
    del backbone,pca;torch.cuda.empty_cache()


def load_record(meta):
    if meta['eval_split']!='confirmation':
        d=old_load(meta);memory=torch.load(OLD/'memories'/f'{meta["id"]}_predicted.pt',map_location='cpu',weights_only=False)['memory']
    else:
        d={key:torch.load(meta[field],map_location='cpu',weights_only=False) for key,field in [('raw','raw'),('rec','tokens'),('target','target')]}
        memory=torch.load(meta['memory'],map_location='cpu',weights_only=False)['memory']
    return dict(d,memory=memory)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','download','extract'))
    globals()[parser.parse_args().stage]()
