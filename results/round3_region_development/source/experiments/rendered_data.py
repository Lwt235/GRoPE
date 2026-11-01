"""Observed patch geometry and point-to-patch correspondence benchmark.

No surface ids, visibility labels, injected error magnitude or true normals are
supplied to the matcher beyond the explicitly named oracle geometry condition.
The depth_fit condition is a shared estimator from noisy rendered depth, not an
RGB-trained foundation model. This distinction is intentional and recorded.
"""
import numpy as np
import torch
from models.geometry import logcov_descriptor, pairwise_squared, project, project_covariance
from experiments.rendered_scene import WIDTH,HEIGHT,PATCH,FOCAL,COLS,ROWS,pair

CONDITIONS=['oracle','depth_noise','normal_noise','scale_noise','mixed','depth_fit','wide']


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def world_rays(x,y,rotation):
    cam=np.stack([(x-WIDTH/2)/FOCAL,(y-HEIGHT/2)/FOCAL,np.ones_like(x)],-1)
    return cam@rotation


def patch_observation(view,cx,cy,rng,condition):
    n=len(cx)
    rotation,translation=view['rotation'],view['translation']
    origin=-rotation.T@translation
    ray=world_rays(cx,cy,rotation)
    true_depth=view['depth'][cy,cx].astype(np.float64)
    normal=view['normal'][cy,cx].astype(np.float64).copy()
    # Fixed stream draws irrespective of condition keep causal controls paired.
    dz=rng.normal(size=n)
    dn=rng.normal(size=(n,3))
    ds=rng.normal(size=(n,2))
    mixed_type=rng.integers(0,4,n)
    depth_sigma=np.zeros(n)
    normal_sigma=np.zeros(n)
    scale_sigma=np.zeros(n)
    if condition=='depth_noise': depth_sigma[:]=.18
    if condition=='normal_noise': normal_sigma[:]=.45
    if condition=='scale_noise': scale_sigma[:]=.65
    if condition=='mixed':
        depth_sigma[mixed_type==1]=.18
        normal_sigma[mixed_type==2]=.45
        scale_sigma[mixed_type==3]=.65
    depth=true_depth+depth_sigma*dz
    normal=unit(normal+normal_sigma[:,None]*dn)
    # A second local estimate from a *noisy observed depth neighborhood*, shared
    # by all methods. Its residual can assess consistency without oracle error.
    offsets=np.array([-4,0,4])
    oy,ox=np.meshgrid(offsets,offsets,indexing='ij')
    nx=cx[:,None]+ox.flatten()
    ny=cy[:,None]+oy.flatten()
    neighbor_depth=view['depth'][ny,nx].astype(np.float64)+rng.normal(0,.03,(n,9))
    neighbor=origin+world_rays(nx,ny,rotation)*neighbor_depth[...,None]
    average=neighbor.mean(1)
    centered=neighbor-average[:,None]
    _,eigenvectors=np.linalg.eigh(np.einsum('npi,npj->nij',centered,centered)/9)
    fit_normal=eigenvectors[:,:,0]
    denominator=(fit_normal*ray).sum(-1)
    fit_z=np.divide(((average-origin)*fit_normal).sum(-1),denominator,
                    out=true_depth.copy(),where=np.abs(denominator)>.08)
    fit_z=np.clip(fit_z,.3,12.)
    if condition=='depth_fit':
        depth=.5*fit_z+.5*(true_depth+.03*dz)
        normal=fit_normal
    mean=origin+ray*depth[:,None]
    residual=np.sqrt(np.mean(np.einsum('npi,ni->np',neighbor-mean[:,None],normal)**2,axis=1))
    # All uncertainty estimates use the observations, not the injected sigma.
    # Disagreement between center prediction and a local depth fit is available.
    sigma=np.clip(np.abs(depth-fit_z)+.03,.01,.6)
    uncertainty=ray[:,:,None]*ray[:,None,:]*sigma[:,None,None]**2
    reference=np.tile(np.array([0.,0.,1.]),(n,1))
    reference[np.abs(normal[:,2])>.85]=[1.,0.,0.]
    tangent1=unit(np.cross(normal,reference))
    tangent2=np.cross(normal,tangent1)
    basis=np.stack([tangent1,tangent2],-1)
    p=mean@rotation.T+translation
    x,y,z=p.T
    jac=np.zeros((n,2,3))
    jac[:,0,0]=FOCAL/z
    jac[:,1,1]=FOCAL/z
    jac[:,0,2]=-FOCAL*x/z**2
    jac[:,1,2]=-FOCAL*y/z**2
    jac=jac@rotation
    aa=jac@basis
    local=np.linalg.inv(aa.transpose(0,2,1)@aa+np.eye(2)*.01)*(PATCH**2/12)
    scale=np.exp(scale_sigma[:,None]*ds)
    local=local*scale[:,:,None]*scale[:,None,:]
    cov=basis@local@basis.transpose(0,2,1)+normal[:,:,None]*normal[:,None,:]*.005**2
    projected=jac@cov@jac.transpose(0,2,1)
    support_error=np.abs(np.log(np.maximum(np.trace(projected,axis1=1,axis2=2),1e-8)/(PATCH**2/6)))
    plane_distance=(normal*mean).sum(-1)
    safe=np.where(np.abs(plane_distance)<.03,np.copysign(.03,plane_distance+1e-12),plane_distance)
    homography=rotation[None]+translation[None,:,None]*normal[:,None,:]/safe[:,None,None]
    # Exact Eq.(1) where well-conditioned; deterministic documented fallback.
    cond=np.linalg.cond(homography)
    fallback=(cond>1e4)|(~np.isfinite(cond))
    homography[fallback]=rotation
    return {'mean':mean,'normal':normal,'cov':cov,'uncertainty':uncertainty,
            'residual':residual,'support_error':support_error,'sigma':sigma,
            'homography':homography,'homography_fallback':fallback.astype(np.float32),
            'depth':depth,'ray':ray,'rotation':rotation,'translation':translation,
            'true_mean':view['points'][cy,cx],'true_normal':view['normal'][cy,cx]}


def patch_features(rgb):
    # Actual local RGB (12x12 -> 6x6 average pool), no latent appearance vectors.
    patches=rgb.reshape(ROWS,PATCH,COLS,PATCH,3).transpose(0,2,1,3,4).reshape(ROWS*COLS,PATCH,PATCH,3)
    return patches.reshape(ROWS*COLS,6,2,6,2,3).mean((2,4)).reshape(ROWS*COLS,-1)


def make_dataset(seed,scenes,condition='oracle',cache=None):
    collected=[]
    stats=[]
    for index in range(scenes):
        scene_seed=seed*1000+index
        wide=condition=='wide'
        cache_key=(scene_seed,wide)
        if cache is not None and cache_key in cache:
            a,b,gt=cache[cache_key]
        else:
            a,b,gt=pair(scene_seed,wide)
            if cache is not None:
                cache[cache_key]=(a,b,gt)
        cx,cy=gt['center_x'],gt['center_y']
        rng=np.random.default_rng(scene_seed+82933)
        qa=patch_observation(a,cx,cy,rng,condition)
        kb=patch_observation(b,cx,cy,rng,condition)
        data={}
        for prefix,obs in [('q',qa),('k',kb)]:
            for key,field in [('m','mean'),('n','normal'),('c','cov'),('u','uncertainty'),('h','homography')]:
                data[key+prefix]=torch.tensor(obs[field],dtype=torch.float32)
        pix=np.stack([cx/PATCH,cy/PATCH],-1)
        data['pixq']=torch.tensor(pix,dtype=torch.float32)
        data['pixk']=torch.tensor(pix,dtype=torch.float32)
        data['q']=torch.tensor(patch_features(a['rgb']),dtype=torch.float32)
        data['k']=torch.tensor(patch_features(b['rgb']),dtype=torch.float32)
        data['labels']=torch.tensor(gt['labels'])
        data['normal']=1-(data['nq']@data['nk'].T).square().clamp(max=1)
        data['center']=pairwise_squared(data['mq'],data['mk'])/.4**2
        eq,ek=torch.linalg.eigvalsh(data['cq']),torch.linalg.eigvalsh(data['ck'])
        aq=(eq[:,-1].clamp_min(1e-10)/eq[:,-2].clamp_min(1e-10)).log()
        ak=(ek[:,-1].clamp_min(1e-10)/ek[:,-2].clamp_min(1e-10)).log()
        data['anisotropy']=(aq[:,None]-ak[None,:]).square()
        data['shape']=pairwise_squared(logcov_descriptor(data['cq']),logcov_descriptor(data['ck']))/6
        r=torch.tensor(a['rotation'])[None]
        t=torch.tensor(a['translation'])[None]
        key_to_q,jk,_=project(data['mk'][None],r,t)
        _,jq,_=project(data['mq'][None],r,t)
        data['k_to_q']=key_to_q[0]*(FOCAL/PATCH)+torch.tensor([WIDTH/2/PATCH,HEIGHT/2/PATCH])
        qcov=project_covariance(jq[0],data['cq'])*(FOCAL/PATCH)**2
        kcov=project_covariance(jk[0],data['ck'])*(FOCAL/PATCH)**2
        data['projected_qcov']=qcov
        data['projected_kcov']=kcov
        # Bhattacharyya coefficient in query coordinates, explicitly a support
        # overlap proxy, not an IoU or a new mathematical contribution.
        avg=(qcov[:,None]+kcov[None,:])/2+torch.eye(2)*1e-5
        delta=data['pixq'][:,None]-data['k_to_q'][None]
        mahal=torch.einsum('nmi,nmij,nmj->nm',delta,torch.linalg.inv(avg),delta)/8
        logdet=torch.linalg.slogdet(avg)[1]/2-(torch.linalg.slogdet(qcov+torch.eye(2)*1e-5)[1][:,None]+torch.linalg.slogdet(kcov+torch.eye(2)*1e-5)[1][None,:])/4
        data['overlap']=torch.exp(-(mahal+logdet).clamp(min=0,max=80))
        data['overlap_constant']=torch.exp(-pairwise_squared(data['pixq'],data['k_to_q'])/(8/12))
        # WorldRoPE Eq.(3)-(9), analytic 3-component depth quadrature. Components
        # preserve total estimated depth variance: .75 between + .25 within.
        wmeans,wcovs=[],[]
        for node in [-np.sqrt(3)*np.sqrt(.75),0.,np.sqrt(3)*np.sqrt(.75)]:
            depth=np.maximum(kb['depth']+node*kb['sigma'],.3)
            origin=-b['rotation'].T@b['translation']
            mean=origin+kb['ray']*depth[:,None]
            camera_cov=np.zeros((len(cx),3,3))
            camera_cov[:,0,0]=depth**2*(PATCH/FOCAL)**2/12
            camera_cov[:,1,1]=depth**2*(PATCH/FOCAL)**2/12
            world_cov=b['rotation'].T[None]@camera_cov@b['rotation'][None]+kb['uncertainty']*.25
            uv,jac,z=project(torch.tensor(mean,dtype=torch.float32)[None],r,t)
            assert z.min()>.05
            wcov=project_covariance(jac[0],torch.tensor(world_cov,dtype=torch.float32))*(FOCAL/PATCH)**2
            wmeans.append(uv[0]*(FOCAL/PATCH)+torch.tensor([WIDTH/2/PATCH,HEIGHT/2/PATCH]))
            wcovs.append(wcov)
        data['world_means']=torch.stack(wmeans,1)
        data['world_covs']=torch.stack(wcovs,1)
        # Inference-available consistency features; no true error/visibility.
        quality_q=torch.tensor(np.clip(qa['residual']/.15,0,10),dtype=torch.float32)
        quality_k=torch.tensor(np.clip(kb['residual']/.15,0,10),dtype=torch.float32)
        support=torch.tensor(np.clip(qa['support_error'][:,None]+kb['support_error'][None,:],0,10),dtype=torch.float32)
        data['gate_inputs']=torch.stack([quality_q[:,None].expand(len(cx),len(cx)),quality_k[None,:].expand(len(cx),len(cx)),data['center'].sqrt().clamp(max=10),support],-1)
        gen=torch.Generator().manual_seed(scene_seed+512)
        data['random_normal']=torch.rand(len(cx),len(cx),generator=gen)
        data['pixel_error']=torch.tensor(np.linalg.norm(gt['target_uv'][:,None,:]-np.stack([cx,cy],-1)[None,:,:],axis=-1),dtype=torch.float32)
        stats.append({'unmatched_fraction':float(np.mean(gt['labels']==len(cx))),
                      'occlusion_or_outside_fraction':float(np.mean(~gt['visible'])),
                      'unrepresentable_visible_fraction':float(np.mean(gt['visible']&~gt['represented'])),
                      'normal_error_degrees':float(np.degrees(np.arccos(np.clip(np.abs((qa['normal']*qa['true_normal']).sum(-1)),0,1))).mean()),
                      'position_error':float(np.linalg.norm(qa['mean']-qa['true_mean'],axis=-1).mean()),
                      'homography_fallback_fraction':float(np.mean(kb['homography_fallback']))})
        collected.append(data)
    stacked={k:torch.stack([d[k] for d in collected]) for k in collected[0]}
    assert all(torch.isfinite(v).all() for v in stacked.values())
    return stacked,stats
