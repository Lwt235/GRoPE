"""Independent CPU no-merge/mass/candidate/pooled-geometry interface checks."""
import os,time,itertools,json,resource
import numpy as np
import torch
from experiments import stream_merge_data as d
from experiments import stream_merge_algorithm as alg

def main():
 start=time.monotonic();torch.set_num_threads(2);p=d.load(d.DATA/'validation/C_first_attention.pt');q=p['q'].double().numpy()/8;a=p['k'].double().numpy();v=p['v'].double().numpy()
 def exact(q,a,v,bias=None):
  L=q@a.transpose(0,2,1)
  if bias is not None:L+=bias
  L-=L.max(-1,keepdims=True);w=np.exp(L);return (w@v)/w.sum(-1,keepdims=True)
 ref=exact(q,a,v);order=np.r_[np.arange(0,5),np.arange(35,len(a[0])),np.arange(5,35)];reorder=exact(q,a[:,order],v[:,order],np.zeros(len(order)));readerror=np.linalg.norm(ref-p['native_y'].double().numpy())/np.linalg.norm(ref)
 # Three identical members and log3 must agree with their full discrete mass.
 smallA=a[:,:12];smallV=v[:,:12];fullA=np.concatenate([smallA[:,:8],np.repeat(smallA[:,8:],3,axis=1)],1);fullV=np.concatenate([smallV[:,:8],np.repeat(smallV[:,8:],3,axis=1)],1);dup=exact(q,fullA,fullV);weighted=exact(q,smallA,smallV,np.r_[np.zeros(8),np.full(4,np.log(3))])
 rng=np.random.default_rng(20261115);f=torch.from_numpy(rng.normal(size=(9,49152)).astype(np.float32));n=torch.tensor([1,2,3,1,2,1,1,3,2]);mu=torch.from_numpy(rng.normal(size=(9,3)));cov=torch.eye(3,dtype=torch.float64)[None].repeat(9,1,1)*.03;m=n.clone();active=torch.ones(9,dtype=torch.bool);C,_=alg.content_matrix(f,n);pairs=list(itertools.combinations(range(9),2));cost={(i,j):n[i].item()*n[j].item()/(n[i].item()+n[j].item())*np.square(f[i].numpy().astype(float)-f[j].numpy().astype(float)).sum() for i,j in pairs};ordered=sorted(pairs,key=lambda ij:(cost[ij],ij));candidate=ordered[:9];checks={};maxrel=max(abs(float(C[i,j])-cost[i,j])/max(cost[i,j],1) for i,j in pairs)
 def geom(i,j,policy):
  mi=int(m[i]);mj=int(m[j]);t=mi+mj;di=mu[i].numpy()-mu[j].numpy();S=(mi*cov[i].numpy()+mj*cov[j].numpy())/t+mi*mj/t**2*np.outer(di,di)
  if policy=='I':return 3*(t*np.log(np.trace(S)/3+.001)-mi*np.log(np.trace(cov[i].numpy())/3+.001)-mj*np.log(np.trace(cov[j].numpy())/3+.001))
  return t*np.linalg.slogdet(S+.001*np.eye(3))[1]-mi*np.linalg.slogdet(cov[i].numpy()+.001*np.eye(3))[1]-mj*np.linalg.slogdet(cov[j].numpy()+.001*np.eye(3))[1]
 for policy in ['C','I','G']:
  i,j,g,why=alg.choose(C,active,m,mu,cov,policy,.001,9);expected=ordered[0] if policy=='C' else min(candidate,key=lambda ij:(geom(*ij,policy),ij));checks[policy]=dict(selected=[i,j],independent_expected=list(expected),agree=(i,j)==expected);assert (i,j)==expected
 m[3]=0;expected=min([ij for ij in pairs if 3 in ij],key=lambda ij:(cost[ij],ij));missing={}
 for policy in ['C','I','G']:
  i,j,_,why=alg.choose(C,active,m,mu,cov,policy,.001,9);assert (i,j)==expected;missing[policy]=[i,j,why]
 zeros=torch.zeros_like(C);zeros.masked_fill_(torch.arange(9)[:,None]>=torch.arange(9)[None,:],float('inf'));m=torch.ones(9,dtype=torch.int64);mu.zero_();tie={}
 for policy in ['C','I','G']:
  i,j,_,_=alg.choose(zeros,active,m,mu,cov,policy,.001,9);assert (i,j)==(0,1);tie[policy]=[i,j]
 first=d.plan()['catalog'][0]['id'];initials={x:d.load(d.OUT/'initials'/f'{x}_{first}.pt') for x in ['C','I','G']};scale=max(float((initials[x]['scale']-initials['C']['scale']).abs().max()) for x in initials);delta=max(abs(initials[x]['delta']-initials['C']['delta']) for x in initials)
 result=dict(complete=True,PID=os.getpid(),CPU_only=True,model_GPU_calls=0,no_merge_reordering_full_denominator_max_abs=float(np.abs(ref-reorder).max()),native_BF16_SDPA_vs_independent_FP64_relative_L2=float(readerror),logn_repeated_member_identity_max_abs=float(np.abs(dup-weighted).max()),all16heads_first6query=True,candidates=checks,direct_current_FP32_dot_vs_literal_FP64_Ward_max_relative_difference=maxrel,missing_geometry_priority=missing,source_ID_ties=tie,matched_first_frame_scale_max_abs=scale,matched_first_frame_delta_max_abs=delta,source_effective_key_FP32_stored_before_native_BF16_read=True,first_C_I_G_completed_prefix_replay={x:d.read(d.OUT/'completed'/f'{x}_{first}.json')['no_merge_or_W_native_replay'] for x in ['C','I','G'] if (d.OUT/'completed'/f'{x}_{first}.json').exists()},W_full_replay=d.read(d.OUT/'completed'/f'W_{first}.json')['no_merge_or_W_native_replay'],seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);d.write(d.OUT/'validation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
