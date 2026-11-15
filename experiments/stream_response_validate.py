"""Independent CPU exact discrete/encoded full-denominator validation, no model calls."""
import os,time,resource,json,struct
import numpy as np
import torch
from experiments import stream_response_data as d

def binary(path):
 with path.open('rb') as f:
  magic,G,H,D,R,*_=struct.unpack('<8s7q',f.read(64));assert magic==b'SRMM001\0';counts=[];means=[];slopes=[]
  for _ in range(G):
   node,n=struct.unpack('<2q',f.read(16));counts.append(n);means.append(np.frombuffer(f.read(d.MEAN),'<f4').reshape(2,H,D).copy())
   if R:slopes.append(np.frombuffer(f.read(d.SLOPE),'<f4').reshape(2,H,D,3).copy())
  assert not f.read(1)
 return np.asarray(counts,np.float64),np.stack(means,2).astype(np.float64),None if not R else np.stack(slopes,2).astype(np.float64)
def main():
 begin=time.monotonic();torch.set_num_threads(2);p=d.load(d.DATA/'validation'/'first_case.pt');r=p['q'].double().numpy()/8.;a=p['a'].double().numpy();v=p['v'].double().numpy();remote=p['remote'].numpy();D=np.ones(a.shape[1],bool);D[remote]=False;L=r@a.transpose(0,2,1)
 def response(L,V):
  mx=L.max(-1);e=np.exp(L-mx[...,None]);Z=e.sum(-1);return mx+np.log(Z),(e@V)/Z[...,None]
 logF,oF=response(L,v);logD,oD=response(L[:,:,D],v[:,D]);ref=[oF,oD];errors={};case=p['case'];W=p['weight'].double().numpy()
 for m in ('X','K','M'):
  counts,means,slopes=binary(d.DATA/'encoded'/f'{case}_{m}.bin');ma,mv=means;g=r@ma.transpose(0,2,1)+np.log(counts)[None,None];s=None
  if slopes is not None:
   B,C=slopes;s=np.einsum('hqd,hgdr->hqgr',r,B);g+=.5*np.square(s).sum(-1)
  mx=np.maximum(g.max(-1),logD);eg=np.exp(g-mx[...,None]);ed=np.exp(logD-mx);Z=ed+eg.sum(-1);out=ed[...,None]*oD+eg@mv
  if slopes is not None:out+=np.einsum('hqgr,hgdr->hqd',eg[...,None]*s,C)
  ref.append(out/Z[...,None])
 for j,label in enumerate(['F','D','X','K','M']):
  gpu=p['GPU_outputs'][j].double().numpy();dif=gpu-ref[j];project=np.einsum('qk,dk->qd',(ref[j]-oF).transpose(1,0,2).reshape(6,1024),W);gpu_project=np.einsum('qk,dk->qd',(gpu-p['GPU_outputs'][0].double().numpy()).transpose(1,0,2).reshape(6,1024),W);den=np.linalg.norm(ref[j]);errors[label]=dict(max_absolute=float(np.abs(dif).max()),relative_L2=float(np.linalg.norm(dif)/den) if den else None,projected_difference_max_absolute=float(np.abs(project-gpu_project).max()))
 # Counted source masses and full denominator, independent zero-query exact identity.
 zmean={}
 for m in ('X','K','M'):
  counts,means,_=binary(d.DATA/'encoded'/f'{case}_{m}.bin');weighted=np.einsum('g,hgd->hd',counts,means[1]);raw=v[:,remote].sum(1);zmean[m]=dict(source_count_sum=int(counts.sum()),exact_remote_tokens=len(remote),zero_query_mean_numerator_max_abs=float(np.abs(weighted-raw).max()),reader_bytes=(d.DATA/'encoded'/f'{case}_{m}.bin').stat().st_size)
 result=dict(complete=True,PID=os.getpid(),case=case,CPU_float64_independent=True,model_GPU_calls=0,all16heads=True,query_IDs=list(range(6)),exact_source_keys=a.shape[1],remote_keys=len(remote),all_D_and_group_mass_combined=True,projection_bias_cancelled=True,errors=errors,zero_query_mass_identity=zmean,seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 d.write(d.OUT/'validation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
