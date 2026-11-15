"""Locked round035 four-cell diagnostic; engineering identities only."""
from experiments import stream_history_data as old
ROOT=old.ROOT;OLDOUT=old.OUT;OLDDATA=old.DATA
OUT=ROOT/'results/gaussian_memory/stream_history_factorization1'
DATA=OLDDATA.parent/'stream_history_factorization_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_035/science_plan.md'
SCIENCE_SHA='937e4d5badb255ada4d8b85bc3d1182f47112438d356e9d6d42c11c0241ac4a5'
now=old.now;sha=old.sha;read=old.read;write=old.write;save=old.save;load=old.load;size=old.size;rgb=old.rgb
def plan():return read(OUT/'configuration.json')
def journal(name,**row):
 import json
 p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**row))+'\n');f.flush()
def prepare():
 assert sha(SCIENCE)==SCIENCE_SHA
 if (OUT/'configuration.json').exists():return plan()
 base=old.plan();OUT.mkdir(exist_ok=True)
 p=dict(experiment='stream_history_factorization1',round=36,science_plan=str(SCIENCE),science_plan_sha256=SCIENCE_SHA,created_utc=now(),catalog=base['catalog'],model_path=base['model_path'],model_sha256_inherited=base['model_sha256_inherited'],official_commit=base['official_commit'],old_configuration=str(OLDOUT/'configuration.json'),old_configuration_sha256=sha(OLDOUT/'configuration.json'),fixed_label_markers={m['id']:dict(path=str(OLDOUT/'label_pairs'/f"{m['id']}.json"),sha256=sha(OLDOUT/'label_pairs'/f"{m['id']}.json")) for m in base['catalog']},cells=['WW','WF','FW','FF'],frames=96,streams=12,camera_iterations=4,dtype=base['dtype'],native_hw=[392,518],bootstrap=base['bootstrap'],near_zero_translation_m=base['near_zero_translation_m'],all_data_used_development=True,training=0,downloads=0,budgets=dict(GPU_worker_hours=4,preflight_cost_hours=.5,DATA_peak_GiB=32,project_peak_GiB=1,worker_VRAM_GiB=40),RGB_readable_batch_backing_bytes=sum(f['rgb_bytes'] for m in base['catalog'] for f in m['frames']),prefix_shared_frames=list(range(9)),expected_unique_aggregator_calls_per_stream=9+87*4,expected_unique_camera_depth_calls_per_stream=9+87*4,preflight='First complete96 stream, reused formally; original WW replay, common-prefix identity, reference immutability, no crossed state commit, measured full-history staging and cost admission before scores. No extra model calls.',staging='Native attention prehook transfers selected immutable CPU rawKV to CUDA; posthook writes only the current rawKV to source-specific CPU file-backed banks. Original attention/SDPA/positions/dtypes unchanged. F source bank is its own causal state; W reference keeps a separate bounded cache, and W shadow is write-only with respect to that reference. Shared mmap backing explicitly counted; no uncharged teacher.',recovery='Cooperative stop after complete frame; flush banks, save exact W KV/common camera/RNG/outputs plus manifest. Immutable past bank rows, next-frame index and sampled tensor identities verified on restore. Abrupt uncheckpointed error halts without replay.',resource_concurrency='Each live stream reserves two full96 rawKV bank files (~18.3GiB) plus bounded recovery/outputs. The32GiB DATA ceiling permits one such active/resumable stream, irrespective of idlecard count; switch to any genuinely safe idle card dynamically.',role=dict(original_creation='gpt-6.1-sol/high',Fast_interface_supported=False,effective_backend_unknown=True))
 write(OUT/'configuration.json',p);return p
if __name__=='__main__':print(dict(prepared=len(prepare()['catalog'])))
