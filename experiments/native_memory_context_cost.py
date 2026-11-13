"""Complete cached-state query cost and independent source-writer timing."""
import time
import numpy as np
import torch
from experiments.native_memory_context_data import *
from experiments.native_receiver_runtime import model_load,forward_read as old_forward
from experiments.native_receiver_data import core_load
from experiments.native_memory_context_train import modules,verify_lock
from experiments.native_memory_context_prepare import source_graph
from models.gaussian_scene_memory import rasterize_field
from models.gaussian_memory_coarse_pe import patch_tokens
from experiments.gaussian_scene_data import query_image

def sync():torch.cuda.synchronize()
@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();verify_lock();marker=OUT/'cost.json'
    if marker.exists():assert read(marker)['complete'];return
    begin=time.monotonic();model=model_load(p,offload_unused=False);sync();cold=time.monotonic()-begin;fusion,writer=modules(SEEDS[0]);records=[];writer_records=[]
    cases=[('fit_00',1,'missing'),('fidelity_00',2,'full')]
    for kind in ['P','S','G','Render']:
        state=load(parent.checkpoint_path('Render',SEEDS[0]) if kind=='Render' else checkpoint_path(kind,SEEDS[0]));core_load(model,state['native']);fusion.load_state_dict(state['model'] if kind=='Render' else state['fusion'])
        if kind!='Render':writer.load_state_dict(state['writer'])
        del state
        for rid,q,c in cases:
            query=load(query_path(rid,q),'cuda');field=load(scene_path(rid) if kind=='Render' else memory_path(kind,SEEDS[0],rid),'cuda');K=query['K'].clone();K[:2]/=14;values=[];segments=[];torch.cuda.reset_peak_memory_stats()
            for iteration in range(25):
                budget_charge('nontraining_forwards');sync();start=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):patches=patch_tokens(model,query_image(query,c))
                sync();encoded=time.monotonic();colors=field['features'].float()*field['has_content'][:,None];rendered,_,_=rasterize_field(field,query['c2w'][None],K[None],32,32,colors);extra=rendered[0].reshape(1024,1024);sync();render_at=time.monotonic();sample=dict(patches=patches,extra=extra,hw=(448,448));depth=old_forward(model,sample,fusion,'Render',checkpoint_blocks=False);sync();end=time.monotonic();assert torch.isfinite(depth).all() and (depth>0).all()
                if iteration>=5:values.append(end-start);segments.append(dict(encoder_seconds=encoded-start,render_seconds=render_at-encoded,native_seconds=end-render_at))
            records.append(dict(kind=kind,id=rid,query=q,condition=c,seed=SEEDS[0],warmup=5,measured=20,seconds=values,segments=segments,peak_GiB=torch.cuda.max_memory_allocated()/2**30,writer_calls=0));print(dict(query_cost_done=kind,id=rid),flush=True)
            if kind!='Render':
                source=load(scene_path(rid),'cuda');sync();t=time.monotonic();graph=source_graph(source);sync();graph_s=time.monotonic()-t;wv=[]
                for iteration in range(25):
                    budget_charge('independent_writer_forwards');sync();t=time.monotonic();encoded=writer(source['features'],graph,kind);sync();elapsed=time.monotonic()-t
                    if iteration>=5:wv.append(elapsed)
                marker=read(OUT/'memories'/f'{kind}_{SEEDS[0]}_{rid}.json');writer_records.append(dict(kind=kind,id=rid,graph_relation_seconds=graph_s,writer_seconds=wv,warmup=5,measured=20,serialization_seconds=marker['serialization_seconds'],reload_seconds=marker['reload_seconds'],persistent_bytes=marker['bytes']))
            del query,field;torch.cuda.empty_cache()
    summary={k:dict(median_ms=1000*float(np.median([v for r in records if r['kind']==k for v in r['seconds']])),peak_GiB=max(r['peak_GiB'] for r in records if r['kind']==k)) for k in ['P','S','G','Render']};amortized={}
    tq=summary['Render']['median_ms']/1000
    for kind in ['P','S','G']:
        wr=[r for r in writer_records if r['kind']==kind];graph=float(np.mean([r['graph_relation_seconds'] for r in wr]));wt=float(np.median([t for r in wr for t in r['writer_seconds']]));ser=float(np.mean([r['serialization_seconds'] for r in wr]));reload=float(np.mean([r['reload_seconds'] for r in wr]));query=summary[kind]['median_ms']/1000
        amortized[kind]=dict(graph_seconds=graph,writer_seconds=wt,serialization_seconds=ser,reload_seconds=reload,write_seconds=graph+wt,totals={str(n):dict(new_seconds=graph+wt+n*query,ratio_to_R=(graph+wt+n*query)/(n*tq),with_serialization_reload_seconds=graph+wt+ser+reload+n*query) for n in [1,4,8,32]})
    write(OUT/'cost.json',dict(complete=True,records=records,writer_records=writer_records,summary=summary,amortization=amortized,total_native_forwards=200,independent_writer_forwards=150,cold_model_load_seconds=cold,source_construction_reference=str(parent.OUT/'cost_description.json'),source_cost_not_zero=True,scope='Preprocessed RGB+reloaded persisted source memory, encoder+actual original renderer+fusion+native+DPT; no writer/source RGB during query. Source graph/writer cost once shown separately, serialization/reload separate. JPEG/disk/camera estimation excluded from warm query.',shared_GPU_not_exclusive_speed_claim=True))
if __name__=='__main__':main()
