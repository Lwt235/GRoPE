"""Bounded checks of sensor boundaries, moments, paired fits and old replay."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_oracle_data import OUT, OLD, CACHE, TARGETS, SEEDS, VARIANTS, checked, load_record, input_seeds, model_path
from models.gaussian_memory_surface import SurfaceMemory


def main():
    plan=checked(True);torch.set_num_threads(4);runs=[];boundaries=[];moment_checks=[]
    for seed in SEEDS:
        for variant in VARIANTS:
            path=model_path(variant,seed);directory=path.parent
            logs=[json.loads(s) for s in (directory/'training.jsonl').read_text().splitlines()]
            old_logs=[json.loads(s) for s in (model_path(variant,seed,'predicted').parent/'training.jsonl').read_text().splitlines()]
            assert len(logs)==2000 and [r['step'] for r in logs]==list(range(1,2001))
            key=lambda r:(r['episode'],r['slot'],r['sample_ids'])
            assert list(map(key,logs))==list(map(key,old_logs)) and np.isfinite([[r['loss'],r['gradient_norm']] for r in logs]).all()
            initial=torch.load(model_path(variant,seed,initial=True),map_location='cpu',weights_only=False)['model']
            old=torch.load(model_path(variant,seed,'predicted',True),map_location='cpu',weights_only=False)['model']
            assert initial.keys()==old.keys() and all(torch.equal(initial[k],old[k]) for k in old)
            status=json.loads((directory/'status.json').read_text());assert status['complete'];runs.append(dict(variant=variant,seed=seed,**status))
            if variant=='fixed':assert all(r['center']==r['shape']==0 for r in status['writer_gradients'])
    training={m['row']['scene'].split('/')[0] for m in plan['catalog'] if m['id'] in plan['training_ids']}
    assert len(training)==12 and not training & set(plan['primary_places'])
    for meta in plan['catalog']:
        d=load_record(meta);sensor,raw,target=d['sensor'],d['raw'],d['target'];pred=input_seeds(d,'predicted','cpu');new=input_seeds(d,'sensor','cpu')
        assert torch.equal(pred['seed_context'],new['seed_context']) and torch.equal(pred['seed_teacher'],new['seed_teacher'])
        assert sensor['reference_frames']==[0,4,8,12] and not sensor['query_depth_used'] and not sensor['query_RGB_used']
        refs={x['path'] for x in sensor['sources']};queries={x['path'] for x in target['sources']};assert refs.isdisjoint(queries)
        assert [x['frame'] for x in sensor['sources']]==[0,4,8,12]
        for s in sensor['sources']:assert sha256(s['path'])==s['sha256']
        boundaries.append({'episode':meta['id'],'reference_query_depth_disjoint':True,'seed_content_equal':True})
        if meta['id'] in ('fit_00','coverage_04','fidelity_07'):
            ref=0;source=sensor['sources'][ref]
            with Image.open(source['path']) as im:encoded=np.asarray(im).astype('uint32')
            metres=((encoded>>3)+((encoded&7)<<13)).astype('float32')/1000
            iy=np.asarray(Image.fromarray(np.arange(metres.shape[0],dtype='int32')[:,None]).resize((1,210),Image.Resampling.NEAREST))[:,0]
            ix=np.asarray(Image.fromarray(np.arange(metres.shape[1],dtype='int32')[None]).resize((280,1),Image.Resampling.NEAREST))[0]
            small=metres[iy[:,None],ix[None]];unit=raw['diagnostics']['unit_meters'];K=raw['K'][0].numpy().astype('float64')*np.array([14,14,1])[:,None];pose=raw['c2w'][0].numpy().astype('float64')
            errors=[]
            for row,col in ((2,2),(7,9),(12,17)):
                yy,xx=np.meshgrid(np.arange(row*14,(row+1)*14)+.5,np.arange(col*14,(col+1)*14)+.5,indexing='ij')
                uv=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3);z=small[row*14:(row+1)*14,col*14:(col+1)*14].flatten()
                valid=np.isfinite(z)&(z>.1)&(z<10);idx=row*20+col
                assert bool(sensor['replace'][idx])==(int(valid.sum())>=98)
                if not valid.any():continue
                points=(uv[valid]@np.linalg.inv(K).T)*(z[valid,None]/unit);points=points@pose[:3,:3].T+pose[:3,3]
                mean=points.mean(0);cov=np.cov(points,rowvar=False,bias=True)+np.eye(3)*.005**2
                a=float(np.max(np.abs(mean-sensor['patch_sensor_mean'][idx].numpy())));b=float(np.max(np.abs(cov-sensor['patch_sensor_cov'][idx].numpy())))
                assert a<2e-5 and b<2e-5;errors.append({'patch':idx,'mean_max_diff':a,'cov_max_diff':b})
            moment_checks.append({'episode':meta['id'],'patches':errors})
    scores=json.loads((OUT/'evaluation/scores.json').read_text());old_scores=json.loads((OLD/'evaluation/scores.json').read_text())
    index={(r['episode'],r['variant'],r['seed'],r['condition']):r for r in old_scores['rows']}
    old_full={(r['episode'],r['variant'],r['seed']):r for r in old_scores['full_query']}
    replay=[]
    for r in scores['rows']:
        if (r['train_geometry']==r['read_geometry']=='predicted' or r['variant']=='local') and r['condition']=='normal':
            old=index[(r['episode'],r['variant'],r['seed'],'normal')]
            replay.append(max(abs(r[k]-old[k]) for k in ('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel')))
    for r in scores['full_query']:
        if r['train_geometry']==r['read_geometry']=='predicted' or r['variant']=='local':
            replay.append(abs(r['log_mse']-old_full[(r['episode'],r['variant'],r['seed'])]['log_mse']))
    assert max(replay)<1e-6
    assert len(scores['rows'])==46*79 and len(scores['full_query'])==46*39
    memory=json.loads((OUT/'memory_identity.json').read_text());assert max(r['replay_max_diff'] for r in memory['state_checks'])==0
    lock=json.loads((OUT/'evaluation_lock.json').read_text())
    for path,digest in lock['models'].items():assert sha256(path)==digest
    independent=[]
    for rid in ('fit_00','transfer_08','fidelity_07'):
        meta=next(m for m in plan['catalog'] if m['id']==rid);d=load_record(meta);raw,rec=d['raw'],d['rec'];sample=rec['roles'].index('evaluation');q=int(rec['query'][sample])
        maps=torch.load(OUT/'evaluation'/(rid+'_maps.pt'),map_location='cpu',weights_only=False)
        for variant in VARIANTS:
            seed=SEEDS[0];name=f'sensor_sensor_{variant}_{seed}';net=SurfaceMemory(variant,plan['local_width'],plan['center_variance']).eval()
            net.load_state_dict(torch.load(model_path(variant,seed),map_location='cpu',weights_only=False)['model']);del net.writer
            saved=torch.load(OUT/'memories'/(rid+'_'+name+'.pt'),map_location='cpu',weights_only=False)['memory']
            with torch.inference_mode():p=net.read(rec['input_tokens'][sample:sample+1],saved,raw['c2w'][q:q+1],raw['K'][q:q+1],raw['diagnostics']['unit_meters'])[0]
            diff=float((p-maps['predictions'][name+'_normal'][0]).abs().max());assert diff<2e-4
            prediction=maps['predictions'][name+'_normal'].numpy().astype('float64');truth=maps['target'].numpy().astype('float64');valid=maps['valid'].numpy()
            mse=float((((prediction-truth)**2*valid).sum(-1)/valid.sum(-1)).mean())
            old=next(r for r in scores['rows'] if r['episode']==rid and r['variant']==variant and r['seed']==seed and r['train_geometry']==r['read_geometry']=='sensor' and r['condition']=='normal')
            err=abs(mse-old['log_mse']);assert err<1e-6
            independent.append({'episode':rid,'variant':variant,'CPU_read_max_difference':diff,'numpy_score_difference':err,'writer_removed':True})
    write_json(OUT/'checks.json',{'complete':True,'new_runs':runs,'boundaries':boundaries,'independent_reference_moments':moment_checks,
               'old_scores_replay_max_diff':max(replay),'independent_reads_scores':independent,'saved_memory_replay_max_diff':0.,
               'source_sha256':sha256(__file__),'primary_training_places_disjoint':True,'initial_and_sampling_identical':True})
    print(json.dumps({'checks_complete':True,'new_runs':len(runs),'old_replay_max_difference':max(replay),'CPU_read_max_difference':max(r['CPU_read_max_difference'] for r in independent)}))


if __name__=='__main__':main()
