"""Frozen R71 models on five previously unused places; inference only."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch

from experiments.round65_common import ROOT,read,write,sha
from experiments import projection_confirmation_data as data_module
from experiments import projection_confirmation_metadata as metadata
from experiments import round70_nvs_run as base
from models.round70_nvs import CompactNVS
from models.round71_target_projection import TargetProjectionNVS,ROLES as PROJECTED_ROLES

OUT=data_module.OUT
ROLES=tuple(PROJECTED_ROLES)+('observed_prope',)
SEEDS=(11,22,33)
CONDITIONS=base.CONDITIONS
R70=ROOT/'results/round70_nvs/attempt1'
R71=ROOT/'results/round71_target_projection/attempt1'
R72=ROOT/'results/round72_rgb_projection/attempt1'
MODULE='experiments.projection_confirmation'


def freeze():
    assert not (OUT/'evaluation_lock.json').exists()
    cohort=metadata.check();complete=read(metadata.OUT/'metadata_complete.json')
    assert complete['selection_sha256']==sha(metadata.RAW/'selection.json')
    assert complete['manifest_sha256']==sha(metadata.RAW/'metadata_manifest.json')
    assert read(R71/'review.json')['scientific_review_complete']
    assert read(R71/'result.json')['decision']['fixed_utility_criterion_met']
    assert read(R72/'review.json')['scientific_review_complete']
    assert not read(R72/'result.json')['decision']['fixed_utility_criterion_met']
    unit=read(OUT/'units.json');assert unit['passed']
    assert not any(metadata.RAW.rglob('*.jpg')) and not any(metadata.RAW.rglob('*.png'))
    sources=dict(read(R71/'lock.json')['source_sha256']);sources.update(cohort['source_sha256'])
    files=[Path(__file__),ROOT/'experiments/projection_confirmation_data.py',ROOT/'experiments/projection_confirmation_report.py',
        ROOT/'tests/test_projection_confirmation.py',OUT/'units.json',metadata.OUT/'cohort_lock.json',
        metadata.OUT/'metadata_complete.json',metadata.RAW/'selection.json',metadata.RAW/'metadata_manifest.json',
        R71/'lock.json',R71/'result.json',R71/'review.json',R71/'audit.json',R72/'result.json',R72/'review.json']
    files.extend(Path(r['path']) if Path(r['path']).is_absolute() else metadata.RAW/r['path']
                 for r in read(metadata.RAW/'metadata_manifest.json'))
    checkpoints={}
    for role in ROLES:
        root=R71 if role in PROJECTED_ROLES else R70
        for seed in SEEDS:
            folder=root/'runs'/role/str(seed);run=read(folder/'run.json');checkpoint=folder/'best.pt'
            assert sha(checkpoint)==run['selected_sha256']
            checkpoints[f'{role}/{seed}']=dict(path=str(checkpoint),sha256=sha(checkpoint),
                selected_epoch=run['selected_epoch'],selected_on='old validation only',training_lock_sha256=sha(root/'lock.json'))
            files.append(folder/'run.json')
    for f in files:sources[str(f)]=sha(f)
    for p,digest in unit['source_sha256'].items():assert sha(p)==digest,p
    for p,digest in sources.items():assert sha(p)==digest,p
    write(OUT/'evaluation_lock.json',dict(groups=['observed'],roles=ROLES,seeds=SEEDS,conditions=CONDITIONS,
        source_sha256=sources,checkpoints=checkpoints,cohort_lock_sha256=sha(metadata.OUT/'cohort_lock.json'),
        scenes=cohort['scenes'],triples=240,pixels_access_permitted=True,created_unix=time.time(),
        hypothesis='R71 target-relative5 retains utility on unused SUN3D places without adaptation.',
        competing_explanation='Repeated development on the old five evaluation scenes inflated the observed advantage.',
        decision='Against each of four fixed controls: base PSNR positive in all three seeds, mean >=0.10 dB, mean LPIPS not worse. Failure limits the claim; no tuning or replacement on these scenes.',
        rgb_group_excluded='R72 did not meet its fixed development criterion; no RGB fresh rescue.',
        implementation=dict(inference='unchanged R70 infer, batch8 bf16',metrics='unchanged R70 metrics',
            preprocessing='R70 frame preprocessing and inherited NVSData.batch; verified on old fit/val',
            gpu_jobs=4,proxy='disabled',python=sys.version,numpy=np.__version__,torch=torch.__version__),
        goal_complete=False))
    (OUT/'evaluation_lock.sha256').write_text(sha(OUT/'evaluation_lock.json')+'\n')
    print('EVALUATION_FROZEN',sha(OUT/'evaluation_lock.json'),'15 frozen models, no new training',flush=True)


def check():
    lock=data_module.check_sources()
    for row in lock['checkpoints'].values():assert sha(row['path'])==row['sha256'],row['path']
    return lock


def export(role,seed):
    lock=check();base.configure();data=data_module.ConfirmationData();net=base.perceptual_metric()
    identity=lock['checkpoints'][f'{role}/{seed}']
    model=(TargetProjectionNVS(role) if role in PROJECTED_ROLES else CompactNVS(role)).cuda()
    model.load_state_dict(torch.load(identity['path'],map_location='cpu',weights_only=False)['model'])
    dest=OUT/'predictions'/role/str(seed);dest.mkdir(parents=True,exist_ok=False)
    for condition in CONDITIONS:
        pred,target=base.infer(model,data,role,condition);values=base.metrics(pred,target,net)
        stored=pred.astype('float16')
        if not np.array_equal(stored.astype('float32'),pred):stored=pred
        path=dest/f'{condition}.npz'
        np.savez_compressed(path,prediction=stored,target_uint8=np.rint(target*255).astype('uint8'),
            scene_id=data.arrays['scene_id'],**values)
        write(path.with_suffix('.json'),dict(role=role,seed=seed,condition=condition,
            checkpoint_sha256=identity['sha256'],evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
            data_manifest_sha256=sha(OUT/'data_manifest.json'),prediction_sha256=sha(path),samples=len(pred)))
        print('EXPORTED',role,seed,condition,flush=True)


def baselines():
    check();base.configure();data=data_module.ConfirmationData();net=base.perceptual_metric()
    dest=OUT/'baselines';dest.mkdir(exist_ok=False)
    for condition in CONDITIONS:
        nearest=[];average=[];targets=[]
        for start in range(0,len(data),8):
            inputs,target=data.batch(np.arange(start,min(start+8,len(data))),'rgb_raymap','cuda',condition)
            distance=(inputs['c2w'][:,:2,:3,3]-inputs['c2w'][:,2:,:3,3]).norm(dim=-1)
            best=distance.argmin(-1)
            nearest.append(inputs['ref_rgb'][torch.arange(len(best),device='cuda'),best].cpu().numpy())
            average.append(inputs['ref_rgb'].mean(1).cpu().numpy());targets.append(target.cpu().numpy())
        target=np.concatenate(targets)
        for name,images in [('nearest_camera_copy',nearest),('reference_average',average)]:
            pred=np.concatenate(images);values=base.metrics(pred,target,net)
            np.savez_compressed(dest/f'{name}_{condition}.npz',prediction=pred,
                target_uint8=np.rint(target*255).astype('uint8'),scene_id=data.arrays['scene_id'],**values)
        print('BASELINES',condition,flush=True)
    write(dest/'identity.json',dict(evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        data_manifest_sha256=sha(OUT/'data_manifest.json'),files={str(f):sha(f) for f in dest.glob('*.npz')}))


def audit():
    check();base.configure();net=base.perceptual_metric();rows=[]
    for role in ROLES:
        for seed in SEEDS:
            path=OUT/'predictions'/role/str(seed)/'base.npz'
            with np.load(path) as z:
                ids=[0,48,96,144,192]
                values=base.metrics(z['prediction'][ids].astype('float32'),z['target_uint8'][ids].astype('float32')/255,net)
                error={k:float(np.max(np.abs(v-z[k][ids]))) for k,v in values.items()}
            assert max(error.values())<2e-5
            rows.append(dict(role=role,seed=seed,sampled_metric_errors=error))
    write(OUT/'audit.json',dict(passed=True,rows=rows,result_sha256=sha(OUT/'result.json'),
        description='Independent primary arithmetic is in report; sampled SSIM/LPIPS replay here. Not external replication.'))


def available_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10);rows=[]
    for line in output.strip().splitlines():
        index,uuid,used,free,util=[x.strip() for x in line.split(',')]
        row=dict(index=int(index),uuid=uuid,memory_mib=int(used),free_mib=int(free),utilization=int(util))
        idle=row['memory_mib']<=512 and row['utilization']<=10
        if row['free_mib']>=12288 and (idle or row['index']==1):
            row['idle_at_observation']=idle;rows.append(row)
    return sorted(rows,key=lambda x:(not x['idle_at_observation'],x['index']))


def queue():
    check();assert not (OUT/'execution_identity.json').exists()
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        maximum_owned_gpu_jobs=4,started_unix=time.time(),user_authorized_parallel=True,goal_complete=False)
    write(OUT/'execution_identity.json',meta);active=[];completed=[];pending=[];stage='initializing'
    def record(state,**extra):
        write(OUT/'queue_status.json',dict(meta,state=state,stage=stage,active=[r[1] for r in active],
            completed=completed,pending=pending,observed_unix=time.time(),**extra))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def batch(action,jobs,gpu=True):
        nonlocal stage,pending
        stage=action;pending=list(jobs)
        while pending or active:
            for process,item,stream in list(active):
                code=process.poll()
                if code is not None:
                    stream.close();active.remove((process,item,stream));completed.append(dict(item,exit_code=code))
                    if code:raise RuntimeError(f"{item['log']}: exit {code}")
            choices=available_gpus() if gpu and pending else [None]
            busy={item['gpu']['uuid'] for _,item,_ in active if item['gpu'] is not None}
            for device in choices:
                if not pending or len(active)>=(4 if gpu else 1):break
                if device is not None and device['uuid'] in busy:continue
                role,seed=pending.pop(0);cmd=[sys.executable,'-m',MODULE,action]
                if role:cmd+=['--role',role]
                if seed is not None:cmd+=['--seed',str(seed)]
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES='' if device is None else device['uuid'])
                log=OUT/f'{action}_{role}_{seed}_{time.time_ns()}.log';stream=log.open('x')
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(pid=process.pid,action=action,role=role,seed=seed,gpu=device,log=str(log),started_unix=time.time())
                active.append((process,item,stream))
                if device is not None:busy.add(device['uuid'])
                print('LAUNCHED',item,flush=True)
            record('running' if active else 'waiting_for_gpu');time.sleep(5)
    try:
        batch('download',[(None,None)],False);batch('prepare',[(None,None)],False)
        batch('export',[(role,seed) for seed in SEEDS for role in ROLES])
        batch('baselines',[(None,None)]);batch('summarize',[(None,None)],False);batch('audit',[(None,None)])
        stage='complete';record('complete',scientific_review_pending=True,visual_review_pending=True)
    except BaseException as error:
        for process,item,stream in active:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            stream.close()
        record('stopped',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','queue','download','prepare','export','baselines','summarize','audit'])
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action in ('download','prepare'):getattr(data_module,args.action)()
    elif args.action=='export':export(args.role,args.seed)
    elif args.action=='summarize':
        from experiments.projection_confirmation_report import summarize
        summarize()
    else:globals()[args.action]()
