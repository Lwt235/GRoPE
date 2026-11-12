"""Old-fit-only exact-input and full native gradient validation of CPU staging."""
import copy
import time

from experiments.learned_projection_data import *
from experiments.learned_projection_stream_train import dataset
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_scene_preflight import objective
from models.gaussian_scene_learned_projection import LearnedProjectionReader
from models.gaussian_scene_native import forward_read


def assert_equal(a,b):
    if torch.is_tensor(a):
        assert a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride() and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b


def reference_sample(p,kind,rid,q,c):
    record=read(OUT/'geometry'/f'{rid}.json');field=load(record['source_path'],'cuda')
    geometry=geometry_tensor(load(geometry_path(rid,q),'cuda'),kind)
    return dict(patches=load(patch_path(rid,q,c),'cuda')['patches'],features=field['features'],
        valid=field['has_content'],geometry=geometry,hw=(448,448),query=load(query_path(rid,q),'cuda'),
        condition=c,place=next(m['place'] for m in p['training_catalog'] if m['id']==rid))


def main():
    torch.set_num_threads(4);p=plan();pilot=copy.deepcopy(p);pilot['training_catalog']=p['training_catalog'][:2]
    model=model_load(p);records=[];torch.cuda.reset_peak_memory_stats()
    for kind in ('S','G'):
        samples,_=dataset(pilot,kind,True)
        for meta in pilot['training_catalog']:
            key=(meta['id'],2,'missing');cached=reference_sample(pilot,kind,*key);staged=samples[key]
            assert_equal(cached,staged)
            reader=LearnedProjectionReader().cuda()
            reader.load_state_dict(load(OUT/f'{kind}_{SEEDS[0]}'/'step_2000.pt')['model'])
            results=[];gradients=[]
            for sample in (cached,staged):
                reader.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind)
                    loss=objective(depth,sample['query']['target'],sample['query']['valid'])
                loss.backward();assert torch.isfinite(loss)
                results.append(dict(loss=float(loss.detach()),depth=depth.detach().cpu()))
                gradients.append({n:v.grad.detach().cpu().clone() for n,v in reader.named_parameters() if v.grad is not None})
                del depth,loss
            max_depth=float((results[0]['depth']-results[1]['depth']).abs().max())
            max_gradient=max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0])
            assert abs(results[0]['loss']-results[1]['loss'])<=1e-5*(1+abs(results[0]['loss']))
            assert torch.allclose(results[0]['depth'],results[1]['depth'],rtol=1e-5,atol=1e-5)
            assert all(torch.isfinite(x).all() for g in gradients for x in g.values())
            records.append(dict(id=meta['id'],query=2,condition='missing',kind=kind,
                all_input_values_dtypes_shapes_strides_exact=True,losses=[r['loss'] for r in results],
                max_abs_depth_difference=max_depth,max_abs_gradient_difference=max_gradient))
            del reader,cached,staged,results,gradients,sample;torch.cuda.empty_cache()
        del samples
    peak=torch.cuda.max_memory_allocated()/2**30
    assert peak<7
    write_json(OUT/'memory_staging/validation.json',dict(complete=True,records=records,peak_allocated_GiB=peak,
        source_sha256=sha256(ROOT/'experiments/learned_projection_stream_train.py'),
        validation_source_sha256=sha256(ROOT/'experiments/learned_projection_stream_check.py'),
        no_optimizer_update=True,no_development_data_used=True,
        scope='Resource implementation validation, four old-fit cases with trained first-seed readers; cached values and all scientific functions unchanged. Not new efficacy evidence.'))
    print(dict(memory_staging_validated=True,records=len(records),peak_GiB=peak),flush=True)


if __name__=='__main__':main()
