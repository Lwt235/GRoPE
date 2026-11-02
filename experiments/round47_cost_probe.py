"""Prospective paired CUDA cost probe; synthetic inputs, no training or selection."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from models.round46_native3d_pe import make_model as make46
from models.round47_conditional_gaussian import make_model as make47
from experiments.round43_frequency_train import ROOT, sha, read, write, checked

OUT=ROOT/'results/round47_conditional/cost_probe1'
R46=ROOT/'results/round46_native3d/attempt1'
R47=ROOT/'results/round47_conditional/attempt1'
ROLES=(('rope2d',46,'center2d',2),('conditional2d',47,'conditional2d',2),
       ('free_quadratic2d',47,'free_quadratic2d',2),('rope3d',46,'point3d',3),
       ('conditional3d',47,'conditional3d',3),('mlp3d',46,'mlp3d',3))


def freeze():
    if OUT.exists():raise ValueError('retain existing probe')
    for root in (R46,R47):
        checked(root)
        if not read(root/'execution_completion.json')['completed']:raise ValueError('completed round required')
    records=[]
    for label,round_,role,dim in ROLES:
        path=(R46 if round_==46 else R47)/'runs'/role/'11'/'best.pt'
        records.append(dict(label=label,round=round_,role=role,dim=dim,path=str(path),sha256=sha(path)))
    OUT.mkdir(parents=True)
    write(OUT/'plan.json',dict(source_sha256=sha(__file__),models=records,sizes=[432,1024],batch=4,
        warmup=5,repeats=20,phases=['inference','forward_backward'],dtype='float32 with frozen float64 PE internals',
        measured='synchronized wall ms and CUDA-event ms; incremental peak allocated above resident models/inputs',
        order='rotate six model order each repeat; all models resident; gradients cleared between calls',
        backward_loss='mean square of final logits; no optimizer',data='synthetic fixed seed4726, no actual labels',
        fresh_data=False,changes_frozen_models=False,selection_rule=None,unix=time.time(),
        limitations='One idle A6000 on a shared host, not a controlled production throughput benchmark or task sequence-length generalization.'))
    (OUT/'plan.sha256').write_text(sha(OUT/'plan.json')+'\n')


def free_devices():
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
    used={row.split(',')[0].strip() for row in apps if ',' in row}
    return [dict(index=int(i),uuid=u) for row in rows for i,u,m,v in [[x.strip() for x in row.split(',')]]
        if u not in used and int(m)<16 and int(v)==0]


def run():
    plan=read(OUT/'plan.json')
    if sha(OUT/'plan.json')!=(OUT/'plan.sha256').read_text().strip() or sha(__file__)!=plan['source_sha256']:
        raise ValueError('plan/source identity')
    if (OUT/'result.json').exists():raise ValueError('retain result')
    first={d['uuid'] for d in free_devices()};time.sleep(1)
    available=[d for d in free_devices() if d['uuid'] in first]
    if not available:raise RuntimeError('no verified idle GPU; keep this plan and retry after inspecting state')
    gpu=available[0]
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    write(OUT/'launch.json',dict(pid=os.getpid(),gpu=gpu,device=torch.cuda.get_device_name(),torch=str(torch.__version__),unix=time.time()))
    models=[]
    for record in plan['models']:
        if sha(record['path'])!=record['sha256']:raise ValueError('checkpoint identity')
        model=(make46 if record['round']==46 else make47)(record['role']).cuda()
        model.load_state_dict(torch.load(record['path'],map_location='cuda:0',weights_only=True)['state_dict'])
        models.append(model)
    rows=[]
    for count in plan['sizes']:
        torch.manual_seed(4726);inputs={}
        common={key:torch.randn(plan['batch'],count,128,device='cuda:0') for key in ('q','k')}
        for dim in (2,3):
            q=torch.randn(plan['batch'],count,dim,device='cuda:0')*6
            k=torch.randn_like(q)*6
            value=dict(common,qmean=q,kmean=k,k_to_q=k,q_to_k=q)
            for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):
                value[key]=torch.zeros(plan['batch'],count,dim,dim,device='cuda:0')
            inputs[dim]=value
        for phase in plan['phases']:
            backward=phase=='forward_backward'
            for iteration in range(-plan['warmup'],plan['repeats']):
                order=list(range(len(models)));offset=(iteration+plan['warmup'])%len(order)
                for index in order[offset:]+order[:offset]:
                    model=models[index];record=plan['models'][index]
                    model.train(backward);model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    wall=time.perf_counter();start.record()
                    with torch.set_grad_enabled(backward):
                        logits=model(inputs[record['dim']])
                        if backward:logits.square().mean().backward()
                    end.record();torch.cuda.synchronize();wall_ms=1000*(time.perf_counter()-wall)
                    cuda_ms=start.elapsed_time(end);peak=torch.cuda.max_memory_allocated()-before
                    if not torch.isfinite(logits).all():raise ValueError('nonfinite probe output')
                    del logits;model.zero_grad(set_to_none=True)
                    if iteration>=0:rows.append(dict(role=record['label'],tokens=count,batch=plan['batch'],
                        phase=phase,repeat=iteration,wall_ms=wall_ms,cuda_ms=cuda_ms,incremental_peak_MiB=peak/2**20))
            print('COST',count,phase,'complete',flush=True)
    summary=[]
    for record in plan['models']:
        for count in plan['sizes']:
            for phase in plan['phases']:
                selected=[r for r in rows if (r['role'],r['tokens'],r['phase'])==(record['label'],count,phase)]
                summary.append(dict(role=record['label'],tokens=count,phase=phase,repeats=len(selected),
                    wall_ms_quantiles=dict(zip(('p10','median','p90'),np.quantile([r['wall_ms'] for r in selected],[.1,.5,.9]).tolist())),
                    median_cuda_ms=float(np.median([r['cuda_ms'] for r in selected])),
                    max_incremental_peak_MiB=max(r['incremental_peak_MiB'] for r in selected)))
    write(OUT/'rows.json',rows)
    write(OUT/'result.json',dict(summary=summary,plan_sha256=sha(OUT/'plan.json'),rows_sha256=sha(OUT/'rows.json'),
        gpu=gpu,limitations=plan['limitations'],fresh_data=False,changes_frozen_models=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);globals()[args.stage]()


if __name__=='__main__':main()
