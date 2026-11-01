"""Frozen RGB-only metric depth and visual features, independent of GT depth.

Camera poses are provided RGB-D reconstruction metadata: this is NOT a fully
RGB-only system. No depth-image path or supervision object enters frame_inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.deps/round5'))
sys.path.insert(0, str(ROOT/'third_party/Depth-Anything-V2/metric_depth'))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DATA = ROOT/'data/round5_sun3d'
CACHE = ROOT/'results/round5_inputs/cache'
SIZE = (192,144)
STRIDE = 8

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def native_cameras(scene):
    kpath=DATA/scene['scene']/'intrinsics.txt'
    ppath=DATA/scene['pose_file']
    if hash_file(kpath)!=scene['K_sha256'] or hash_file(ppath)!=scene['pose_sha256']:
        raise ValueError('camera metadata changed after frame selection')
    K = np.loadtxt(kpath)
    K[:2,2] -= 1  # Native SUN3D toolbox uses 1-based pixel centers.
    source = np.loadtxt(ppath).reshape(-1,3,4)
    poses = np.broadcast_to(np.eye(4),(len(source),4,4)).copy()
    poses[:,:3] = source
    return K,poses

def resize_K(K, original_size):
    out=K.copy()
    scale=np.array(SIZE)/np.array(original_size)
    out[:2,:2] *= scale[:,None]
    out[:2,2] = (out[:2,2]+.5)*scale-.5
    return out

def support_from_prediction(predicted_depth, K):
    """Uniform 3D moments of each 8x8 predicted-depth patch, no GT mask.

    This yields one Gaussian support per RGB token, not optimized 3DGS R/S.
    Clamping is fixed, image-only and applied before any supervision is read.
    """
    z=np.nan_to_num(predicted_depth,nan=10.,posinf=20.,neginf=.1).clip(.1,20.)
    yy,xx=np.mgrid[:SIZE[1],:SIZE[0]]
    xyz=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1)
    patches=xyz.reshape(18,8,24,8,3).transpose(0,2,1,3,4).reshape(432,64,3)
    mu=patches.mean(1)
    centered=patches-mu[:,None]
    cov=np.einsum('nsi,nsj->nij',centered,centered)/64
    return mu.astype('float32'),cov.astype('float32')

def projected_support(mean,cov,source_c2w,target_c2w,target_K):
    """First-order projection in token-stride units, fixed finite fallback."""
    transform=np.linalg.inv(target_c2w)@source_c2w
    m=mean@transform[:3,:3].T+transform[:3,3]
    c=transform[:3,:3]@cov@transform[:3,:3].T
    z=m[:,2].clip(.1,None)
    J=np.zeros((len(m),2,3),dtype=np.float64)
    J[:,0,0]=target_K[0,0]/z/STRIDE
    J[:,1,1]=target_K[1,1]/z/STRIDE
    J[:,0,2]=-target_K[0,0]*m[:,0]/z**2/STRIDE
    J[:,1,2]=-target_K[1,1]*m[:,1]/z**2/STRIDE
    xy=(m[:,:2]/z[:,None]*[target_K[0,0],target_K[1,1]]+target_K[:2,2])/STRIDE
    C=J@c@J.transpose(0,2,1)
    vals,vecs=np.linalg.eigh((C+C.transpose(0,2,1))/2)
    clipped=(vecs*vals.clip(0,256)[:,None,:])@vecs.transpose(0,2,1)
    diagnostics={'behind_or_near':int((m[:,2]<.1).sum()),
                 'cov_clipped':int((vals.max(-1)>256).sum()),
                 'mean_clipped':int((np.abs(xy)>128).any(-1).sum())}
    return xy.clip(-128,128).astype('float32'),clipped.astype('float32'),diagnostics

@torch.inference_mode()
def frame_inputs(rgb_path,K,c2w,depth_model,backbone,device):
    import cv2
    # The only image opened in this function is RGB.
    with Image.open(rgb_path) as im:
        raw=np.array(im.convert('RGB'))
        small=np.array(im.convert('RGB').resize(SIZE,Image.Resampling.BOX))
        resized_K=resize_K(K,im.size)
    pred=depth_model.infer_image(np.ascontiguousarray(raw[:,:,::-1]),input_size=518)
    pred=cv2.resize(pred,SIZE,interpolation=cv2.INTER_AREA)
    mu,cov=support_from_prediction(pred,resized_K)
    x=torch.from_numpy(small.copy()).to(device).permute(2,0,1).float().unsqueeze(0)/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    x=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
    x=backbone.layer2(backbone.layer1(x))
    # ResNet layer2 centers are at input 0,8,...; align to patch 3.5,11.5,...
    yy,xx=torch.meshgrid(torch.arange(18,device=device),torch.arange(24,device=device),indexing='ij')
    grid=torch.stack([2*(xx+.4375+.5)/24-1,2*(yy+.4375+.5)/18-1],-1)[None]
    feat=F.grid_sample(x,grid,align_corners=False,padding_mode='border').flatten(2).transpose(1,2)
    feat=F.normalize(feat,dim=-1).squeeze(0).cpu().numpy()
    return {'features':feat,'mean3':mu,'cov3':cov,'predicted_depth':pred.astype('float32'),
            'K':resized_K.astype('float64'),'c2w':c2w.astype('float64')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['train','val']);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int)
    args=p.parse_args();torch.set_num_threads(4)
    from depth_anything_v2.dpt import DepthAnythingV2
    from experiments.real_frozen_features import official_backbone
    weights=ROOT/'results/round5_inputs/downloads/depth_anything_v2_metric_hypersim_vits.pth'
    depth=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    depth.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    depth.eval().requires_grad_(False).to(args.device)
    backbone,backbone_meta=official_backbone(Path('/home/liwt/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'))
    backbone.to(args.device)
    producer={'source_sha256':hash_file(__file__),'depth_weights_sha256':hash_file(weights),
              'backbone_weights_sha256':backbone_meta['weights_sha256'],
              'geometry_version':2}
    selection=json.loads((DATA/'selection.json').read_text());done=0;start=time.time()
    manifest=[]
    for scene in selection['scenes']:
        if scene['split'] not in args.splits:continue
        K,poses=native_cameras(scene)
        for frame in scene['frames']:
            rgb=DATA/scene['scene']/'image'/frame['rgb']
            dest=CACHE/scene['scene']/f"{frame['index']:07d}.npz"
            dest.parent.mkdir(parents=True,exist_ok=True)
            provenance=dict(producer,rgb_sha256=hash_file(rgb),K_sha256=scene['K_sha256'],
                            pose_sha256=scene['pose_sha256'],original_index=frame['index'])
            sidecar=dest.with_suffix('.json')
            verified=False
            if dest.exists() and sidecar.exists():
                previous=json.loads(sidecar.read_text())
                verified=previous.get('producer')==provenance and previous.get('cache_sha256')==hash_file(dest)
            if not verified:
                arrays=frame_inputs(rgb,K,poses[frame['index']],depth,backbone,args.device)
                np.savez_compressed(dest,**arrays)
                sidecar.write_text(json.dumps({'producer':provenance,'cache_sha256':hash_file(dest)},indent=2))
            manifest.append({'scene':scene['scene'],'split':scene['split'],'index':frame['index'],
                             'rgb_sha256':hash_file(rgb),'cache_sha256':hash_file(dest)})
            done+=1
            if done%20==0:print('CACHED',done,'elapsed',round(time.time()-start,1),flush=True)
            if args.limit and done>=args.limit:break
        if args.limit and done>=args.limit:break
    result={'frames':manifest,'selection_sha256':hash_file(DATA/'selection.json'),
            'depth_weights_sha256':hash_file(weights),'depth_source_commit':'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf',
            'backbone':backbone_meta,'geometry':'uniform 8x8 RGB-predicted-depth 3D moments, no primitive internal covariance',
            'camera_condition':'provided RGB-D reconstructed camera poses, shared by all methods',
            'max_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.time()-start,
            'source_sha256':hash_file(__file__)}
    dest=ROOT/'results/round5_inputs'/('cache_'+'_'.join(args.splits)+('_smoke' if args.limit else '')+'.json')
    dest.write_text(json.dumps(result,indent=2));print('DONE',done,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
