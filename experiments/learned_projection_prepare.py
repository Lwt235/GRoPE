"""Reuse source fields, prepare two PE variants, and check only old training inputs."""
import argparse
import time

from experiments.learned_projection_data import *
from experiments.gaussian_scene_config import schedule
from experiments.gaussian_scene_preflight import objective, official_current
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_scene_memory import center_ray_descriptor, contribution_weights, SceneReader
from models.gaussian_scene_native import forward_read
from models.gaussian_scene_learned_projection import LearnedProjectionReader, spherical_field


@torch.no_grad()
def prepare_one(meta):
    rid=meta['id']; marker=OUT/'geometry'/f'{rid}.json'
    if marker.exists():
        old=read(marker)
        assert old['complete'] and all(sha256(k)==v for k,v in old['files'].items())
        return
    source=SOURCE_CACHE/rid/'scene_memory.pt'; source_hash=sha256(source)
    binding=read(SOURCE_OUT/'binding'/f'{rid}.json')
    assert binding['files'][str(source)]==source_hash and binding['source_only']
    field=load(source,'cuda'); sphere=spherical_field(field)
    assert torch.allclose(field['scales'].float().square().sum(-1),sphere['scales'].square().sum(-1),rtol=1e-5)
    parent=read(SOURCE_OUT/'query_cache'/f'{rid}.json'); files={}; inputs={}; timings=[]
    for q in QUERIES:
        query=load(query_path(rid,q),'cuda'); K=query['K'].clone();K[:2]/=14
        inputs[str(query_path(rid,q))]=sha256(query_path(rid,q))
        original=SOURCE_CACHE/rid/f'geometry_{q}.pt'
        assert sha256(original)==parent['files'][str(original)]
        weights=load(original,'cuda')['weights']
        torch.cuda.synchronize();begin=time.monotonic()
        center=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit'])
        sw,_=contribution_weights(sphere,query['c2w'][None],K[None],32,32)
        sw=sw[0].reshape(1024,-1)
        payload=dict(center=center.bfloat16(),full=torch.log1p(2048*weights.float()).bfloat16(),
                     sphere=torch.log1p(2048*sw.float()).bfloat16())
        assert payload['center'].shape==(1024,2048,7)
        assert all(torch.isfinite(x).all() for x in payload.values())
        torch.cuda.synchronize();timings.append(dict(query=q,seconds=time.monotonic()-begin))
        path=geometry_path(rid,q);path.parent.mkdir(parents=True,exist_ok=True)
        save_tensor(path,{k:v.cpu() for k,v in payload.items()});files[str(path)]=sha256(path)
        for condition in CONDITIONS:
            path=patch_path(rid,q,condition)
            assert sha256(path)==parent['files'][str(path)]
            inputs[str(path)]=parent['files'][str(path)]
    assert sha256(source)==source_hash
    write_json(marker,dict(complete=True,id=rid,place=meta['place'],source_path=str(source),
        source_sha256=source_hash,source_only_field_reused=True,source_content_unchanged=True,
        source_sensor_privilege=True,files=files,inputs=inputs,times=timings))
    print(dict(geometry_complete=rid),flush=True)


def preflight(model,p):
    records=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];field=load(SOURCE_CACHE/rid/'scene_memory.pt','cuda')
        query=load(query_path(rid,2),'cuda'); reference=official_current(model,query_image(query,'missing'))
        payload=load(geometry_path(rid,2),'cuda');patches=load(patch_path(rid,2,'missing'),'cuda')['patches']
        for kind in KINDS:
            torch.manual_seed(SEEDS[0]);reader=LearnedProjectionReader().cuda()
            torch.manual_seed(SEEDS[0]);old=SceneReader(center_pe=True).cuda()
            common=reader.state_dict();previous=old.state_dict()
            assert all(torch.equal(common[k],v) for k,v in previous.items() if k!='center_pe.0.weight')
            assert torch.equal(common['center_pe.0.weight'][:,:7],previous['center_pe.0.weight'])
            assert torch.count_nonzero(common['center_pe.0.weight'][:,7])==0
            del old
            sample=dict(patches=patches,features=field['features'],valid=field['has_content'],
                        geometry=geometry_tensor(payload,kind),hw=(448,448))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                initial=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            error=float((initial-reference).abs().max());assert error<1e-4
            optimizer=torch.optim.AdamW(reader.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);steps=[]
            for step in range(3):
                optimizer.zero_grad(set_to_none=True);trace={}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind,trace=trace)
                    loss=objective(depth,query['target'],query['valid'])
                loss.backward()
                assert torch.isfinite(loss) and all(trace.get(str(i),0)>0 for i in p['read_layers'])
                extra=float(reader.center_pe[0].weight.grad[:,7].float().norm())
                if step==2 and kind!='P':assert extra>0
                if kind=='P':assert extra==0
                norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),p['gradient_clip'])
                assert torch.isfinite(norm);optimizer.step()
                steps.append(dict(step=step+1,loss=float(loss),extra_column_gradient=extra,read_gradients=trace))
            records.append(dict(id=rid,kind=kind,initial_native_max_abs=error,common_initialization_exact=True,steps=steps))
            del reader,optimizer,sample,depth,loss,initial
        del field,query,payload,patches,reference;torch.cuda.empty_cache()
    write_json(OUT/'preflight.json',dict(complete=True,records=records,
        no_development_queries_used=True,pilot_weights_discarded=True,parameters=1054672))


def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);p=initialize_plan()
    rows=p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    for meta in rows:prepare_one(meta)
    if args.pilot:
        preflight(model_load(p),p)
        names=['models/gaussian_scene_learned_projection.py','models/gaussian_scene_memory.py','models/gaussian_scene_native.py',
               'experiments/learned_projection_data.py','experiments/learned_projection_prepare.py','experiments/learned_projection_train.py',
               'experiments/gaussian_scene_train.py','experiments/gaussian_scene_config.py']
        write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(OUT/'plan.json'),
            preflight_sha256=sha256(OUT/'preflight.json'),source_sha256={n:sha256(ROOT/n) for n in names},
            fixed_before_development_outputs=True,formal_training_started=False))
    else:
        write_json(OUT/'geometry_complete.json',dict(complete=True,fields=len(rows),queries_per_field=4,
            plan_sha256=sha256(OUT/'plan.json'),source_fields_not_rebuilt=True,content_not_rebound=True))


if __name__=='__main__':main()
