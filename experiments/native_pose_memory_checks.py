"""Necessary CPU checks of coordinate convention and shared structured transport."""
import sys
import numpy as np
import torch
sys.path.insert(0,'third_party/streamvggt/src')
from experiments.native_pose_memory_data import *
from models.native_pose_memory_reader import transport
from models.region_hrope_control import homographies

def main():
    torch.set_num_threads(4);torch.manual_seed(20261113)
    pose=torch.tensor([[.2,-.1,.3,0.,0.,0.,1.],[-.4,.2,.1,0.,0.,1.,0.]])
    a=dict(rotation=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]]*2),translation=torch.tensor([[1.,2.,3.],[4.,5.,6.]]),scale=torch.tensor([2.,3.]))
    c=mappose(pose,a).numpy();expected=[]
    for i in range(2):
        R=np.eye(3) if i==0 else np.diag([-1.,-1.,1.]);A=a['rotation'][i].numpy();e=np.eye(4);e[:3,:3]=A@R.T;e[:3,3]=a['scale'][i].item()*A@(-R.T@pose[i,:3].numpy())+a['translation'][i].numpy();expected.append(e)
    transform_error=float(np.max(np.abs(c-np.array(expected))));assert transform_error<1e-6
    raw=torch.randn(2,7,13)*.1;raw[...,:6]+=torch.tensor([1.,0.,0.,0.,1.,0.]);raw[...,11]+=1;K=torch.tensor([[[.8,0.,.5],[0.,.9,.5],[0.,0.,1.]]]*2)[:,None]
    H,inv=homographies(raw,K);matrix_error=float((H@inv-torch.eye(3)).abs().max());assert matrix_error<1e-5
    v=torch.randn(2,4,7,48);xy=torch.rand(2,7,2)-.5;roundtrip=transport(transport(v,inv,xy,True),H,xy,False);transport_error=float((v-roundtrip).abs().max());assert transport_error<1e-5
    config=dict(science_contract=dict(path=str(SCIENCE),sha256=sha(SCIENCE)),experiment='native_pose_memory_pe1',CLI_UUID='01a1d350-d1ec-70f2-9e37-8960b2dfe7a8',requested_configuration=dict(model='gpt-6.1-sol',reasoning_effort='high',service_tier='fast',source='explicit coordinator dispatch; not independent backend attestation'),backend_effective=dict(model='unknown',reasoning_effort='unknown',service_tier='unknown'),reader=dict(width=192,heads=4,head_width=48,position_MLP=[29,64,192],Q_input=2069,K_ordinary=1044,homography=13,output_MLP=[384,192,2048],query_features='last aggregator frame-attention component (1024D), source-prefix frozen',zero_initialized=['position last','camera-token residual last'],HRoPE='region_hrope_control homographies and 8x3 + 12xy pairs, theta10000, ell1..6'),training=dict(kinds=KINDS,seeds=SEEDS,steps=2000,batch=2,formal_updates=24000,formal_sample_reader_camera_head=48000,optimizer='AdamW',lr=.0003,weight_decay=.01,clip=5,final_only=True,source_unit='saved source GS unit',loss='squared map center/unit + squared geodesic radians'),expected=dict(source_native=176,query_native_initial=352,predictions=4224,GS_public_states=396,initial_baselines=352),budgets=dict(GPU_worker_hours=8,DATA_GiB=24,project_GiB=2,workers=2),storage=dict(DATA=str(CACHE),original_inputs='read-only references',aggregator_prefix='ephemeral per scene, rebuild from unchanged four RGB sources; accounted bytes/time',camera_prefix='saved FP32 per scene, reused without query cache',checkpoint_every=100,checkpoint='model/Adam/torch+CUDA RNG/numpy sampler/counters, final step2000'),engineering_revision='v1',runtime_sources={p:sha(ROOT/p) for p in ['models/native_pose_memory_reader.py','experiments/native_pose_memory_data.py','experiments/native_pose_memory_run.py','experiments/native_pose_memory_launch.py','experiments/native_pose_memory_checks.py']})
    write(OUT/'configuration.json',config);write(OUT/'coordinate_transport_checks.json',dict(complete=True,coordinate_max_abs=transform_error,homography_inverse_max_abs=matrix_error,transport_roundtrip_max_abs=transport_error,official_pose_quaternion_order='XYZW',GS_quaternion_order='WXYZ',CPU_only=True))
    inherited=read(ROOT/'results/gaussian_memory/native_memory_context_pe1/configuration_lock.json')
    paths={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for m in metas() for p in [source(m['id']),pair(m['id']),scene(m['id'])]+[query(m['id'],q) for q in QUERIES]}
    write(OUT/'input_manifest.json',dict(inputs=paths,identity_authorities=['native_training_coverage1/plan.json and inputs/source_predictions/source_construction records','gaussian_scene_memory1/inputs','source_geometry_control1/inputs','native_memory_context_pe1/preparation'],catalog=dict(path=str(old.OUT/'plan.json'),sha256=sha(old.OUT/'plan.json')),weights=dict(path=plan()['model_path'],bytes=Path(plan()['model_path']).stat().st_size,sha256_inherited=inherited['pretrained_sha256']),source_ids=[0,3,6,9],fit_queries=list(QUERIES[:4]),other_queries=list(QUERIES[4:]),fresh=False))
    print('CPU checks/config/input references complete')
if __name__=='__main__':main()
