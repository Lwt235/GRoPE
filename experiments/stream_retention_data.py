"""Round033 contract engineering and inherited input/label identities."""
from pathlib import Path
from experiments import stream_history_data as old
ROOT=old.ROOT;OUT=ROOT/'results/gaussian_memory/stream_native_retention1';DATA=old.DATA.parent/'stream_native_retention_v1';OLDOUT=old.OUT;OLDDATA=old.DATA
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_033/science_plan.md';SCIENCE_SHA='d0ae8daeac00fe57765e8bfee27e4e470c6339357a0cf59f571190aa9c3af9cc'
now=old.now;sha=old.sha;read=old.read;write=old.write;save=old.save;load=old.load;size=old.size;rgb=old.rgb
METADATA_ARENA_BYTES=128*1024
SERIALIZATION_ALLOWANCE_BYTES=1024*1024
def plan():return read(OUT/'configuration.json')
def journal(name,**x):
 import json
 p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**x))+'\n');f.flush()
def prepare():
 assert sha(SCIENCE)==SCIENCE_SHA
 if (OUT/'configuration.json').exists():return plan()
 base=old.plan();OUT.mkdir(exist_ok=True)
 p=dict(experiment='stream_native_retention1',round=34,science_plan=str(SCIENCE),science_plan_sha256=SCIENCE_SHA,created_utc=now(),catalog=base['catalog'],model_path=base['model_path'],model_sha256_inherited=base['model_sha256_inherited'],official_commit=base['official_commit'],old_configuration=str(OLDOUT/'configuration.json'),old_configuration_sha256=sha(OLDOUT/'configuration.json'),fixed_label_markers={m['id']:dict(path=str(OLDOUT/'label_pairs'/f"{m['id']}.json"),sha256=sha(OLDOUT/'label_pairs'/f"{m['id']}.json")) for m in base['catalog']},policies=['W','T','D'],formal_new_policies=['T','D'],frames=96,streams=12,camera_iterations=4,dtype=base['dtype'],native_hw=[392,518],W_replay_fixed_stream='fidelity_00',preflight_fixed_frames=8,bootstrap=base['bootstrap'],near_zero_translation_m=base['near_zero_translation_m'],all_data_used_development=True,training=0,downloads=0,budgets=dict(GPU_worker_hours=2,preflight_cost_hours=.25,DATA_peak_GiB=8,project_peak_GiB=1,worker_VRAM_GiB=40),metadata_arena_bytes=METADATA_ARENA_BYTES,serialization_allowance_bytes=SERIALIZATION_ALLOWANCE_BYTES,RGB_readable_batch_backing_bytes=sum(f['rgb_bytes'] for m in base['catalog'] for f in m['frames']),T_objective='Continuous sampled-index interval[0,c], exact doubled covering radius max(adjacent retained gaps,2*(c-last)); frame0 fixed. Enumerate 5choose4, lexicographic retained-ID tie.',D_objective='Min over 5choose4 of max candidate cosine-distance to kept remote+frame0+recent3. FP32 native single-image normalized-patch mean, L2 normalized; CPU FP64 distance reduction of these descriptors, lexicographic tie.',recovery='Cooperative exit checkpoint only: after complete frame/eviction, save exactly current retained rawKV/camera/descriptor/RNG/partial outputs, then exit. Restore verifies samples and deletes consumed checkpoint before next frame. No old live checkpoint keeps evicted frames. Abrupt uncheckpointed crash is preserved as failure, never silently replayed.',role=dict(original_creation='gpt-6.1-sol/high',Fast_interface_supported=False,effective_backend_unknown=True))
 write(OUT/'configuration.json',p);return p
if __name__=='__main__':print(dict(prepared=len(prepare()['catalog'])))
