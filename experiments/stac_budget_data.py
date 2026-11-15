"""Round049 engineering identities and fixed input index; no GT evaluation here."""
import json, hashlib, os, time
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/gaussian_memory/stac_native_budget1'
AUTHOR=OUT/'author'
DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/stac_native_budget1')
BASE=ROOT/'results/gaussian_memory/stream_history_value1'
OLDOUT=BASE
OLDDATA=DATA.parent/'stream_history_value_v1'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_048/science_plan.md'
SHA='09a6c033f2442a917f50b3c12791b3500687a9b993d0265d4ca9d89fc83f12ec'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file()) if Path(p).exists() else 0
def plan():return read(OUT/'configuration.json')
def journal(name,**x):
 p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**x),allow_nan=False)+'\n');f.flush()
def prepare():
 assert sha(SCIENCE)==SHA
 if (OUT/'configuration.json').exists():return plan()
 old=read(BASE/'configuration.json')
 # Mechanical index transfer: omit GT poses/intrinsics from the GPU package.
 catalog=[{k:m[k] for k in ('id','place','scene','frames','native_hw')} for m in old['catalog']]
 p=dict(experiment='stac_native_budget1',round=49,science_plan=str(SCIENCE),science_plan_sha256=SHA,
        author_commit='fd7e718597cf9963de85c8fffae32a698e8619f5',catalog=catalog,
        existing_model_path=old['model_path'],model_sha256_inherited=old['model_sha256_inherited'],official_commit=old['official_commit'],
        original_input_configuration=str(BASE/'configuration.json'),original_input_configuration_sha256=sha(BASE/'configuration.json'),
        all_data_used=True,frames=96,streams=12,kf_every=1,chunk_size=4,recent_size=4,hh_size=2,pinned_frame_indices=[0],
        retrieval_size=2,return_buf=True,voxel_size=.05,voxel_num=4096,voxel_buf_cap=8,voxel_piv_cap=4,
        sim_threshold=.8,replace_threshold=.8,score_threshold=.2,allocator='segment',seg_size=1,
        attn_backend='cuda',subsample_ratio=1.,temperature=.9,dtype='BF16 backbone/effective KV; FP32 depth/point/camera heads',
        camera='All arrived prefix pose tokens; four iterations; snapshot at target chunk submission; no final-96 retrospective pose scoring',
        execution_order='S all12 costs/predictions without GT, then seal common B, then maximal legal H/W, then score',
        compressed_readable_all12_RGB_bytes=sum(f['rgb_bytes'] for m in catalog for f in m['frames']),
        history_hard_max=16*2**30,B_quantum=64*2**20,
        budgets=dict(GPU_seconds=28800,preflight_GPU_seconds=3600,source_build_mixed_seconds=5400,DATA_bytes=48*2**30,project_bytes=2*2**30,worker_VRAM_bytes=40*2**30),
        lifecycle='One live exact state; on yield serialize direct device storages + outputs + C++ full owned tensor/config/scalar state; resume consumes old backing before new forward. All actual overlapping disk/GPU/CPU copies count; no permanent hidden token archive.',
        build_deadline_utc='2026-11-15T01:03:31+00:00',GT_scoring_allowed=False)
 write(OUT/'configuration.json',p);DATA.mkdir(parents=True,exist_ok=True);return p
if __name__=='__main__':print(dict(prepared=len(prepare()['catalog'])))
