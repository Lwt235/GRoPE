"""Independent CPU content/member binding check, on two old training cases."""
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *


def main():
    p=plan();rows=[]
    for rid in p['training_ids'][:2]:
        old=load(PARENT_CACHE/rid/'q11_persistent.pt')
        for n in CAPACITIES:
            s=load(sample_path(rid,11,'persistent',n));member=s['membership'].numpy();nsp=int(s['source_special'].sum());ntokens=len(old['query_means']) if 'query_means' in old else old['tokens_per_frame']
            chosen=sorted(set([0,n//4,n//2,n-1,int(s['member_counts'].argmin()),int(s['member_counts'].argmax())]))
            for layer,index in enumerate((20,21,22,23)):
                values=old['memory'][layer]['v'].float().numpy();special=np.arange(values.shape[-2])%ntokens<5
                # Explicit index_select keeps the head axis before the token axis.
                sp=torch.from_numpy(np.flatnonzero(special));expected_special=torch.from_numpy(values).index_select(2,sp)
                actual_special=s['values'][index].index_select(2,torch.arange(nsp)).float()
                assert torch.equal(expected_special,actual_special)
                patch=values[...,~special,:]
                expected=np.stack([patch[:,:,member==j,:].astype(np.float64).mean(2) for j in chosen],2)
                expected=torch.from_numpy(expected.astype(np.float32)).to(torch.bfloat16).float().numpy()
                actual=s['values'][index].index_select(2,torch.tensor([nsp+j for j in chosen])).float().numpy()
                delta=actual-expected;rms=float(np.sqrt(np.mean(delta**2))/max(np.sqrt(np.mean(expected**2)),1e-12));maximum=float(abs(delta).max())
                assert rms<.002,(rid,n,index,rms,maximum)
                rows.append(dict(id=rid,capacity=n,layer=index,slots=chosen,max_abs=maximum,relative_rms=rms,special_values_exact=True))
    write_json(OUT/'independent_content_checks.json',dict(complete=True,rows=rows,scope='Independent float64 member means, BF16 storage; source V and special-token ordering, old training only',plan_sha256=sha256(OUT/'plan.json')))
    print({'complete':True,'checks':len(rows),'max_relative_rms':max(r['relative_rms'] for r in rows)},flush=True)


if __name__=='__main__':main()
