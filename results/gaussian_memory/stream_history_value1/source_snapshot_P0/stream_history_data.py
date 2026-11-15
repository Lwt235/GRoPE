"""Fixed round031 contract inputs; no model, labels, or downloads in selection."""
import json,hashlib,os,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from PIL import Image
ROOT=Path('/home/liwt/GRoPE')
OUT=ROOT/'results/gaussian_memory/stream_history_value1'
DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/stream_history_value_v1')
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_031/science_plan.md'
SCIENCE_SHA='68d3bd4f6cf94429763076267dab69581a4790a4bea067d22762cb5251a821f8'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');torch.save(x,q);q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def journal(name,**x):
 p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(utc=now(),**x))+'\n');f.flush()
def plan():return read(OUT/'configuration.json')
def rgb(frame):
 with Image.open(frame['rgb_path']) as im:
  im=im.convert('RGB');w,h=im.size;shape=(518,round(h*518/w/14)*14);assert shape==(518,392)
  return torch.from_numpy(np.asarray(im.resize(shape,Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255
def sensor(frame):
 from experiments.round5_rgbd_data import load_sensor_depth
 a=load_sensor_depth(frame['depth_path']);a=np.asarray(Image.fromarray(a).resize((280,210),Image.Resampling.NEAREST)).copy();return a,np.isfinite(a)&(a>.1)&(a<10)
def size(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) if Path(p).exists() else 0
def prepare():
 from experiments.round5_rgbd_data import read_sun3d_intrinsics,read_sun3d_extrinsics,resize_intrinsics,sun3d_timestamp
 assert sha(SCIENCE)==SCIENCE_SHA
 if (OUT/'configuration.json').exists():return plan()
 original=read(ROOT/'results/gaussian_memory/native_history1/plan.json');selected={};selection_ids={}
 for name in ['round6_sun3d/selection.json','round6_sun3d/selection_v2.json','round7_sun3d/selection.json']:
  p=ROOT/'data'/name;selection_ids[str(p)]=sha(p)
  for r in read(p)['scenes']:selected[r['scene']]=(p.parent,r)
 catalog=[]
 for meta in original['catalog']:
  root,row=selected[meta['row']['scene']];scene=root/row['scene'];available=[]
  for f in sorted(row['frames'],key=lambda f:(sun3d_timestamp(f['rgb']),f['index'])):
   rp=scene/'image'/f['rgb'];dp=scene/'depth'/f['depth']
   if rp.is_file() and dp.is_file():available.append((f,rp,dp))
  assert len(available)>=96,(row['scene'],len(available))
  indices=np.rint(np.linspace(0,len(available)-1,96)).astype(int);assert len(set(indices))==96
  frames=[];poses=read_sun3d_extrinsics(root/row['pose_file']);K=read_sun3d_intrinsics(scene/'intrinsics.txt')
  for j in indices:
   f,rp,dp=available[j];assert f['index']<len(poses)
   frames.append(dict(index=f['index'],timestamp_us=sun3d_timestamp(f['rgb']),depth_timestamp_us=sun3d_timestamp(f['depth']),rgb_path=str(rp),depth_path=str(dp),rgb_bytes=rp.stat().st_size,depth_bytes=dp.stat().st_size,rgb_mtime_ns=rp.stat().st_mtime_ns,depth_mtime_ns=dp.stat().st_mtime_ns))
  assert len({f['rgb_path'] for f in frames})==len({f['depth_path'] for f in frames})==96
  with Image.open(frames[0]['rgb_path']) as im:hw=(im.height,im.width)
  catalog.append(dict(id=meta['id'],place=meta['place'],scene=row['scene'],available_matched_frames=len(available),uniform_available_indices=indices.tolist(),frames=frames,raw_pose_path=str(root/row['pose_file']),raw_pose_sha256=sha(root/row['pose_file']),intrinsics_path=str(scene/'intrinsics.txt'),intrinsics_sha256=sha(scene/'intrinsics.txt'),c2w_metric=poses[[f['index'] for f in frames]].tolist(),K_metric=resize_intrinsics(K,hw,(210,280)).tolist(),native_hw=[392,518]))
 setup=read(ROOT/'results/gaussian_memory/native_history1/setup.json');assert Path(setup['model_path']).stat().st_size==5026743569
 result=dict(experiment='stream_history_value1',round=32,science_plan=str(SCIENCE),science_plan_sha256=SCIENCE_SHA,created_utc=now(),catalog=catalog,selection_metadata_sha256=selection_ids,selection_rule='All12previously-used places;96unique RGB-D pairs uniformly spaced in timestamp-ordered available matched inventory, before any support/prediction inspection',model_path=setup['model_path'],model_sha256_inherited=setup['model_sha256'],weight_not_rehashed=True,official_commit=setup['github_commit'],arms=['F','W','A'],dtype='BF16 native aggregator, FP32 camera and depth heads; original SDPA backend',camera_iterations=4,history_rule='W aggregator and W/A camera: frame0 plus most recent7 complete frames; every camera refinement retained within current frame; prune only after all4 rounds',formal_streams=12,frames_per_stream=96,unique_aggregator_calls=2304,formal_camera_head_calls=3456,formal_depth_head_calls=2304,profiling_stream=catalog[0]['id'],profiling_outputs_reused_as_first_formal_stream=True,preflight_fixed_frames=3,preflight_native_frame_calls=9,preflight_direct_attention_calls=2,checkpoint_chunk_frames=32,bootstrap=dict(draws=5000,seed=20261114,unit='paired place'),near_zero_translation_m=1e-6,training=0,downloads=0,all_data_used_development=True,budgets=dict(GPU_worker_hours=4,profiling_hours=.5,DATA_peak_GiB=16,project_peak_GiB=1,single_worker_peak_VRAM_GiB=40),state_checkpoint_policy='One active stream; immutable32-frame full-aggregator delta chunks plus latest bounded W/camera states and prediction partials. Only owned resumable chunks removed after atomic final prediction+completion. No CPU offload to rescue VRAM.',role=dict(original_creation='gpt-6.1-sol/high',Fast_parameter_supported=False,effective_backend_unknown=True))
 write(OUT/'configuration.json',result);return result
if __name__=='__main__':print(json.dumps(dict(prepared=len(prepare()['catalog']))))
