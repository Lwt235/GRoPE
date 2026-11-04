"""Read-only input geometry interpretation, no learned outcome or selection."""
from pathlib import Path

import numpy as np
import torch

from models.round70_nvs import PATCH, HEIGHT, WIDTH
from models.round71_target_projection import project_reference
from experiments.round70_nvs_data import NVSData
from experiments.round65_common import read, write, sha
from experiments.round71_target_projection import OUT, OLD, CONDITIONS


def main():
    torch.set_num_threads(4)
    dest=OUT/'input_geometry_diagnostic';plan=read(dest/'plan.json');rows=[]
    for split in ('fit','val'):
        data=NVSData(split)
        for condition in CONDITIONS:
            size=[];ratio=[];total=covered=valid_count=inside=0
            for start in range(0,len(data),8):
                inp,_=data.batch(np.arange(start,min(start+8,len(data))),'observed_prope','cpu',condition)
                r=inp['c2w'][:,:2,:3,:3];t=inp['c2w'][:,:2,:3,3]
                m=torch.einsum('bvij,bvnj->bvni',r,inp['ref_mean'])+t[:,:,None]
                s=r[:,:,None]@inp['ref_cov']@r[:,:,None].transpose(-1,-2)
                coverage=inp['ref_coverage'].flatten(1,2)
                u,c,valid=project_reference(m.flatten(1,2),s.flatten(1,2),inp['K'],inp['c2w'],coverage)
                scale=c.new_tensor([np.sqrt(12)*WIDTH/PATCH,np.sqrt(12)*HEIGHT/PATCH]).double()
                normalized=c.double()*scale[:,None]*scale[None,:]
                eig=torch.linalg.eigvalsh(normalized)[valid]
                assert torch.isfinite(eig).all() and (eig>0).all()
                total+=valid.numel();covered+=int((coverage>0).sum());valid_count+=int(valid.sum())
                inside+=int((valid&((u>=0)&(u<=1)).all(-1)).sum())
                size.append(eig.numpy());ratio.append((eig[:,1]/eig[:,0]).numpy())
            vals=np.concatenate(size);ratios=np.concatenate(ratio)
            rows.append(dict(split=split,condition=condition,triplets=len(data),reference_regions=total,
                covered_regions=covered,positive_depth_and_covered=valid_count,
                active_fraction=valid_count/total,inside_fraction_among_active=inside/valid_count,
                normalized_cov_eigenvalues_quantiles_05_50_95=np.quantile(vals,[.05,.5,.95],axis=0).tolist(),
                normalized_anisotropy_quantiles_05_50_95=np.quantile(ratios,[.05,.5,.95]).tolist()))
    write(dest/'result.json',dict(rows=rows,source_sha256=sha(Path(__file__)),plan_sha256=sha(dest/'plan.json'),
        round71_lock_sha256=sha(OUT/'lock.json'),data_sha256={s:sha(OLD/'data'/f'{s}.npz') for s in ('fit','val')},
        prediction_or_outcome_used=False,configuration_changed=False,new_success_gate=False))
    for row in rows:print(row,flush=True)


if __name__=='__main__':main()
