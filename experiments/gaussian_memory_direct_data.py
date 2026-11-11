"""Reference-only high-dimensional states over explicitly used cohorts."""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_data import CACHE, REPORT, REF, source_geometry
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import OUT as VIEW_OUT, TOKEN_CACHE, CONF_CACHE, load_backbone
from experiments.gaussian_memory_fidelity_data import OUT as FID_OUT, NEW_CACHE as FID_CACHE, checked as fidelity_checked

OUT = ROOT / 'results/gaussian_memory/direct_reference1'
REF_CACHE = DATA / 'direct_reference_v1'
SEEDS = (20261110,20261111,20261112)
VARIANTS = ('local','center','wide','direct')
NEW_VARIANTS = ('wide','direct')


def model_path(variant, seed, initial=False):
    base = FID_OUT / f'B_{variant}_{seed}' if variant in ('local','center') else OUT / f'{variant}_{seed}'
    return base / ('initial.pt' if initial else 'step_2000.pt')


def catalog():
    rows=[]
    originals=json.loads((REPORT / 'data_manifest.json').read_text())['episodes']
    for row in originals:
        rid=row['id']
        rows.append({'id':rid,'row':row,'cohort':'development','raw':str(CACHE / (rid+'.pt')),
                     'tokens':str(TOKEN_CACHE / (rid+'.pt')),
                     'memory':str(REPORT / 'learned_geometry/memories' / (rid+'.pt'))})
    for cohort, root, cache, memdir in [('used_clip40',VIEW_OUT,CONF_CACHE,VIEW_OUT / 'transfer_memories'),
                                       ('used_clip80',FID_OUT,FID_CACHE,FID_OUT / 'memories')]:
        identity=json.loads((root / 'transfer_cache_identity.json').read_text())
        for row in identity['episodes']:
            rid=row['id']
            rows.append({'id':rid,'row':row,'cohort':cohort,'raw':str(cache / (rid+'_raw.pt')),
                         'tokens':str(cache / (rid+'.pt')),'memory':str(memdir / (rid+'.pt'))})
    assert len(rows)==42 and len({r['id'] for r in rows})==42
    return rows


def prepare():
    fidelity_checked(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    sources=['models/gaussian_memory_direct_reference.py','models/gaussian_memory_view_token.py',
             'models/gaussian_memory_attention.py','models/gaussian_memory_projection.py',
             'experiments/gaussian_memory_direct_data.py','experiments/gaussian_memory_direct_train.py',
             'experiments/gaussian_memory_direct_evaluate.py','experiments/gaussian_memory_data.py',
             'experiments/gaussian_memory_view_data.py']
    rows=catalog()
    paths=[model_path(v,s,ini) for v in ('local','center') for s in SEEDS for ini in (False,True)]
    paths += [VIEW_OUT / 'plan.json',FID_OUT / 'plan.json',REPORT / 'data_manifest.json',
              FID_OUT / 'development_evaluation/scores.json',FID_OUT / 'transfer_evaluation/scores.json']
    paths += [r[k] for r in rows for k in ('raw','tokens','memory')]
    plan={'source_sha256':{p:sha256(ROOT / p) for p in sources},
          'plan_sha256':sha256(ROOT / 'docs/gaussian_memory_direct_reference_plan.md'),
          'input_sha256':{str(p):sha256(p) for p in paths},'catalog':rows,
          'seeds':SEEDS,'new_runs':6,'reused_runs':6,'steps':2000,'batch':2,
          'feature_normalization':'fixed per-token LayerNorm then half storage',
          'reference_slots':1200,'feature_dim':2048,'center_variance':.7146816253662109,
          'all_evaluation_data_previously_used':True,'confirmation_claim':False}
    write_json(OUT / 'plan.json',plan)
    print(json.dumps({'prepared':True,'episodes':len(rows),'new_runs':6}),flush=True)


def checked(check_inputs=False):
    plan=json.loads((OUT / 'plan.json').read_text())
    for p,h in plan['source_sha256'].items(): assert sha256(ROOT / p)==h,p
    assert sha256(ROOT / 'docs/gaussian_memory_direct_reference_plan.md')==plan['plan_sha256']
    if check_inputs:
        for p,h in plan['input_sha256'].items(): assert sha256(p)==h,p
    return plan


def load_record(row, device='cpu', with_direct=True):
    raw=torch.load(row['raw'],weights_only=False,map_location='cpu')
    rec=torch.load(row['tokens'],weights_only=False,map_location='cpu')
    memory=torch.load(row['memory'],weights_only=False,map_location=device)['memory']
    result={'row':row['row'],'meta':row,'raw':raw,'rec':rec,'memory':memory}
    if with_direct:
        saved=torch.load(REF_CACHE / (row['id']+'.pt'),weights_only=False,map_location=device)
        result['direct_memory']={k:v.float() for k,v in saved['memory'].items()}
    return result


@torch.inference_mode()
def reference_memory(model, images, K_reference, c2w_reference_metric, old_unit):
    """Only four reference observations enter this function; no query inputs."""
    assert images.shape==(4,3,210,280) and K_reference.shape==(4,3,3)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        tokens,start=model.aggregator(images[None])
    context=tokens[-1][0,:,start:].float().reshape(1200,2048)
    pose=model.camera_head(tokens)[-1]
    depth,confidence=model.depth_head(tokens,images=images[None],patch_start_idx=start)
    # The legacy helper indexes REF in a 13-view tensor; fill all other slots
    # with the first reference, not with query calibration or observations.
    K=K_reference[:1].expand(13,-1,-1).clone(); K[REF]=K_reference
    c2w=c2w_reference_metric[:1].expand(13,-1,-1).clone(); c2w[REF]=c2w_reference_metric
    means,cov,valid,_,diag=source_geometry(depth,confidence,pose,K,c2w)
    ratio=diag['unit_meters']/old_unit
    memory={'means':(means*ratio).cpu(),'covars':(cov*ratio**2).cpu(),
            'opacities':valid.float().cpu(),'features':F.layer_norm(context,(2048,)).half().cpu()}
    assert all(torch.isfinite(v).all() for v in memory.values())
    return memory,dict(diag,coordinate_unit_meters=old_unit,renormalization_ratio=ratio,
                       feature_mean_abs=float(memory['features'].float().mean(-1).abs().max()))


def extract():
    plan=checked(True); torch.set_num_threads(4)
    REF_CACHE.mkdir(parents=True,exist_ok=True)
    model=load_backbone(); started=time.monotonic(); hashes={}; stats=[]
    for row in plan['catalog']:
        raw=torch.load(row['raw'],weights_only=False,map_location='cpu')
        ref_images=raw['rgb'][REF].permute(0,3,1,2).float().cuda()/255.
        K=raw['K'][REF].cuda()*torch.tensor([14,14,1],device='cuda')[:,None]
        poses=torch.tensor(raw['episode']['c2w_metric'],device='cuda').float()[REF]
        memory,diag=reference_memory(model,ref_images,K,poses,raw['diagnostics']['unit_meters'])
        path=REF_CACHE / (row['id']+'.pt'); assert not path.exists()
        torch.save({'memory':memory,'diagnostics':diag,'episode':row['id'],
                    'reference_frames':REF,'source_raw_sha256':sha256(row['raw']),
                    'query_images_used':False,'teacher_used':False},path)
        hashes[str(path)]=sha256(path)
        gs=torch.load(row['memory'],weights_only=False,map_location='cpu')['memory']
        stats.append({'id':row['id'],'slots':len(memory['means']),'valid_slots':int(memory['opacities'].sum()),
                      'direct_tensor_bytes':sum(v.numel()*v.element_size() for v in memory.values()),
                      'GS_tensor_bytes':sum(v.numel()*v.element_size() for v in gs.values()),'diagnostics':diag})
        print(json.dumps({'cached':row['id'],'seconds':time.monotonic()-started,
                          'peak_GiB':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
    write_json(OUT / 'reference_cache_identity.json',{'sha256':hashes,'records':stats,
               'seconds':time.monotonic()-started,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3})


def checked_cache():
    plan=checked(True)
    identity=json.loads((OUT / 'reference_cache_identity.json').read_text())
    for p,h in identity['sha256'].items(): assert sha256(p)==h,p
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','extract'])
    a=p.parse_args();{'prepare':prepare,'extract':extract}[a.command]()
