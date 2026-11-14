import json
"""Round028 identities and new artifacts; immutable round026 inputs stay in place."""
from experiments import native_track_memory_data as prior
from experiments.native_track_memory_data import ROOT,old,read,write,save,load,sha,cpu,device,now,plan,metas,case,anchorfile,point_grid,source,pair,source_patches,fields_path
OUT=ROOT/'results/gaussian_memory/native_source_resolved_pe1'
CACHE=old.DATA/'native_source_resolved_pe_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_028/science_plan.md'
SCIENCE_SHA='19235eca9476843f1ac4965928dbbfe9c9c5c4068f11ae3ddbfaaf89f9e3dec4'
KINDS=('F_P','F_S','F_G','M_P','M_G');SEEDS=prior.SEEDS;SOURCES=prior.SOURCES;QUERIES=prior.QUERIES
def journal(name,**kw):
    import json
    p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**kw))+'\n');f.flush()
def state(rid,representation):return CACHE/'states'/f'{rid}_{representation}.pt'
def init_config():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);assert sha(SCIENCE)==SCIENCE_SHA
    for n in ('workers','logs','worker_resources','calls','binding','fits','completed','evaluation','engineering_revisions'): (OUT/n).mkdir(exist_ok=True)
    if not (OUT/'configuration.json').exists():
        cfg=read(prior.OUT/'configuration.json');cfg.update(science_sha256=SCIENCE_SHA,kinds=KINDS,steps=1000,limits=dict(GPU_worker_hours=18,DATA_GiB=8,project_GiB=1),representation='Source-only train32 PCA255+mean; F=(z_iv,p_iv), M=(p_iv*sum_v z_iv,p_iv); four256-blocks, source0/3/6/9; FP16 persistent content, FP32 decoder/Adam, BF16 native',inherited_inputs=str(prior.OUT/'input_identities.json'),inherited_labels_references_and_geometry=True,inherited_predictions=['P','G','L','Native2','Native4'],no_inherited_prediction_reruns=True)
        write(OUT/'configuration.json',cfg)
