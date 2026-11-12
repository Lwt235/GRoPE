"""Cheap full-cache boundary/storage inventory; no method scores or selection."""
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *


def main():
    p=plan();verify(p);rows=[]
    for key in example_keys(p,'train')+example_keys(p,'development'):
        rid,q,c,n=key;path=sample_path(*key);s=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        sources=s['source_ids'];nsp=len(sources)*5
        assert all(source<q for source in sources) and s['query']==q
        assert s['keys'].shape==s['values'].shape==(24,1,16,n+nsp,64)
        assert s['keys'].dtype==torch.float32 and s['values'].dtype==torch.bfloat16
        assert len(s['camera_state'])==4 and all(t.shape[2]==4*len(sources) for pair in s['camera_state'] for t in pair)
        assert not any(name in s for name in ('prefix','early','memory'))
        assert s['patch_tokens'].shape==(1,1036,1024)
        assert s['query_means'].shape==(1041,3) and s['query_covariances'].shape==(1041,3,3)
        assert s['source_special'].sum()==nsp and s['member_counts'].sum()==len(sources)*1036
        assert np.allclose(s['log_mass'][nsp:].numpy(),np.log(s['member_counts'].numpy()),atol=1e-6)
        fields=('keys','values','source_means','source_covariances','source_special','log_mass','member_counts')
        actual=sum(s[f].numel()*s[f].element_size() for f in fields)+sum(t.numel()*t.element_size() for pair in s['camera_state'] for t in pair)
        record=read(OUT/'cache'/(keyname(*key)+'.json'));assert record['persistent_bytes']==actual
        source_cov=s['source_covariances'][nsp:].numpy().astype(np.float64);query_cov=s['query_covariances'][5:].numpy().astype(np.float64)
        assert np.isfinite(source_cov).all() and np.isfinite(query_cov).all()
        se=np.linalg.eigvalsh(source_cov);qe=np.linalg.eigvalsh(query_cov)
        rows.append(dict(key=list(key),source_ids=sources,persistent_bytes=actual,key_dtype=str(s['keys'].dtype),value_dtype=str(s['values'].dtype),source_min_eigenvalue=float(se[:,0].min()),query_min_eigenvalue=float(qe[:,0].min()),source_nonpositive_eigenvalues=int((se<=0).sum()),query_nonpositive_eigenvalues=int((qe<=0).sum())))
    train_places={m['place'] for m in p['training_catalog']};dev_places={m['place'] for m in p['catalog']};assert not train_places&dev_places
    result=dict(complete=True,records=len(rows),train_records=192,development_records=48,source_boundary_and_shape_checks=True,persistent_bytes_independently_counted=True,train_development_place_disjoint=True,query_rgb_backbone_only_no_fused_prefix=True,all_data_previously_used=True,new_confirmation=False,source_nonpositive_eigenvalues=sum(r['source_nonpositive_eigenvalues'] for r in rows),query_nonpositive_eigenvalues=sum(r['query_nonpositive_eigenvalues'] for r in rows),rows=rows,plan_sha256=sha256(OUT/'plan.json'),source_sha256=sha256(ROOT/'experiments/gaussian_memory_coarse_inventory.py'),scope='Full input/storage inventory; eigenvalues descriptive, not an effect-based gate or utility claim')
    write_json(OUT/'input_inventory.json',result)
    print({k:result[k] for k in ('records','source_nonpositive_eigenvalues','query_nonpositive_eigenvalues','train_development_place_disjoint')},flush=True)


if __name__=='__main__':main()
