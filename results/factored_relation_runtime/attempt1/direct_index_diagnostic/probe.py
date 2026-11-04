import copy,os,time
from pathlib import Path
import numpy as np
import torch
from experiments.factored_relation_runtime import checked,validation_cache,make_model
from experiments import pe_runtime_diagnostic as prior
from models.factored_relation_runtime import factorize
from experiments.round65_common import read,write,sha
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/factored_relation_runtime/attempt1/direct_index_diagnostic'
plan=read(OUT/'plan.json');assert sha(__file__)==plan['source_sha256']
lock=checked();uuid=os.environ['CUDA_VISIBLE_DEVICES']
row=next(r.split(',') for r in prior.gpu_snapshot().splitlines() if r.split(',')[1].strip()==uuid)
assert int(row[2])<=512 and int(row[3])>=34*1024 and int(row[4])<=10
prior.native_engine.configure();assert torch.cuda.is_available() and torch.cuda.device_count()==1
cache=validation_cache('region');identity=lock['checkpoints']['region/direct_mlp']
a=make_model('region','direct_mlp');a.load_state_dict(torch.load(identity['path'],map_location='cpu',weights_only=True)['state_dict']);b=factorize(copy.deepcopy(a),'region')
models=[a.cuda().eval(),b.cuda().eval()];outputs=[];metrics=[]
for model in models:
    x,m=prior.region_engine.evaluate(model,cache);outputs.append(torch.from_numpy(x));metrics.append(m)
np.savez(OUT/'logits.npz',original=outputs[0].numpy(),factored=outputs[1].numpy(),ids=cache['ids'])
world=torch.from_numpy(cache['world_centers']).double();overlap=torch.from_numpy(cache['overlap']).double();rows=[]
for direction in range(2):
    x,y=[v if direction==0 else v.transpose(-1,-2) for v in outputs]
    truth=overlap if direction==0 else overlap.transpose(-1,-2)
    ia,ib=x.argmax(-1),y.argmax(-1)
    for pair,query in (ia!=ib).nonzero().tolist():
        ka,kb=int(ia[pair,query]),int(ib[pair,query]);view=1-direction
        da=float((world[pair,direction,query]-world[pair,view,ka]).norm());db=float((world[pair,direction,query]-world[pair,view,kb]).norm())
        oa,ob=float(truth[pair,query,ka]),float(truth[pair,query,kb]);valid=bool(truth[pair,query].max()>=.3)
        same_inputs=all(np.array_equal(cache[k][pair,view,ka],cache[k][pair,view,kb]) for k in ['features','centers','covariance'])
        rows.append(dict(pair=pair,id=str(cache['ids'][pair]),direction=direction,query=query,old_key=ka,new_key=kb,valid_query=valid,
            old_logit_margin=float(x[pair,query,ka]-x[pair,query,kb]),new_logit_margin=float(y[pair,query,kb]-y[pair,query,ka]),
            identical_key_inputs=same_inputs,world_center_max_difference=float((world[pair,view,ka]-world[pair,view,kb]).abs().max()),
            whole_label_columns_equal=bool(torch.equal(truth[pair,:,ka],truth[pair,:,kb])),
            overlap_old=oa,overlap_new=ob,center_error_old=da,center_error_new=db,
            region_decision_changed=bool((oa>=.3)!=(ob>=.3)),center_decision_changed=bool((da<=.1)!=(db<=.1))))
result=dict(completed_unix=time.time(),source_sha256=sha(__file__),plan_sha256=sha(OUT/'plan.json'),checkpoint_sha256=identity['sha256'],
    changed_queries_by_direction=[sum(r['direction']==d for r in rows) for d in range(2)],changes=rows,
    metrics_original=metrics[0],metrics_factored=metrics[1],metric_differences={k:metrics[1][k]-metrics[0][k] for k in metrics[0]},
    original_mechanical_screen_unchanged=True,model_or_threshold_tuning=False,new_data_access=False,
    logits_sha256=sha(OUT/'logits.npz'),goal_complete=False)
write(OUT/'result.json',result)
print('INDEX_DIAGNOSTIC_COMPLETE',result['changed_queries_by_direction'],result['metric_differences'],flush=True)
