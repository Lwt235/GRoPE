"""Fixed source-only spatial/PCA moment formats and full-denominator response."""
import time,struct
import numpy as np
import torch
from experiments import stream_response_data as d

def sync():torch.cuda.synchronize()
def tree(x,coarse=256,fine=d.MLEAVES):
 # Source ID is the stable tie breaker; refinement never sees queries/features.
 x=np.asarray(x,np.float32)
 if len(x)==0:return {},{}
 leaves={1:np.arange(len(x),dtype=np.int64)};snap=None
 for target in range(1,min(fine,len(x))+1):
  if target==min(coarse,len(x)):snap={k:v.copy() for k,v in sorted(leaves.items())}
  if target==min(fine,len(x)):break
  node=min((k for k in leaves if len(leaves[k])>1),key=lambda k:(-len(leaves[k]),k));ids=leaves.pop(node);axis=int(np.argmax(np.ptp(x[ids],axis=0)));ids=ids[np.lexsort((ids,x[ids,axis]))];mid=len(ids)//2;leaves[node*2]=ids[:mid];leaves[node*2+1]=ids[mid:]
 return snap,dict(sorted(leaves.items()))
def padded(groups,device):
 ids=list(groups);n=np.array([len(groups[k]) for k in ids],np.int64);ix=np.zeros((len(ids),int(n.max())),np.int64);mask=np.zeros_like(ix,dtype=bool)
 for j,k in enumerate(ids):ix[j,:n[j]]=groups[k];mask[j,:n[j]]=True
 return torch.from_numpy(ix).to(device),torch.from_numpy(mask).to(device),torch.from_numpy(n).to(device),np.asarray(ids,np.int64)
def eigroot(cov,dim,rank=None):
 lam,u=torch.linalg.eigh(cov);rank=dim if rank is None else rank;lam=lam[:,-rank:];u=u[:,:,-rank:];tol=dim*torch.finfo(cov.dtype).eps*lam[:,-1:].clamp_min(0);good=lam>tol;root=lam.clamp_min(0).sqrt()*good;inv=torch.where(good,root.clamp_min(torch.finfo(cov.dtype).tiny).reciprocal(),torch.zeros_like(root));return u,root,inv,good

def fit(a,v,xyz,coarse,fine):
 """a/v actual effective source keys/native values [head,source,64]."""
 begin=time.monotonic();device=a.device
 if not coarse:
  means=torch.empty((2,16,0,64),device=device);counts=torch.empty(0,dtype=torch.int64,device=device);nodes=np.empty(0,np.int64);states={m:dict(means=means,counts=counts,nodes=nodes,slopes=None if m=='M' else torch.empty((2,16,0,64,3),device=device)) for m in ('X','K','M')}
  return states,dict(seconds=time.monotonic()-begin,source_tokens=0,coarse_groups=0,fine_groups=0,geometry_rank=torch.empty(0,dtype=torch.int64),geometry_mean=torch.empty((0,3)),geometry_covariance=torch.empty((0,3,3)),key_rank=torch.empty((16,0),dtype=torch.int64),explained={m:[] for m in ('X','K','M')},source_precision='FP64 common;encodingFP32',group_source_indices={},fine_group_source_indices={},all_invalid_exact_common_fallback=True)
 ix,mask,n,nodes=padded(coarse,device);jx,jmask,jn,jnodes=padded(fine,device);G=len(n);J=len(jn);w=mask.double();jw=jmask.double();nn=n.double()[:,None,None];jnn=jn.double()[:,None,None]
 xx=xyz[ix].double();mu=(xx*w[...,None]).sum(1)/n[:,None];xc=(xx-mu[:,None])*w[...,None];cov=xc.transpose(1,2)@xc/nn;u,root,inv,good=eigroot(cov,3);z=xc@(u*inv[:,None]);means=torch.empty((2,16,G,64),device=device);slopes={m:torch.empty((2,16,G,64,3),device=device) for m in ('X','K')};meansM=torch.empty((2,16,J,64),device=device);explained={m:[] for m in ('X','K','M')};keyranks=[]
 for h in range(16):
  aa=a[h,ix].double();vv=v[h,ix].double();ma=(aa*w[...,None]).sum(1)/n[:,None];mv=(vv*w[...,None]).sum(1)/n[:,None];ac=(aa-ma[:,None])*w[...,None];vc=(vv-mv[:,None])*w[...,None];means[0,h]=ma;means[1,h]=mv
  bx=ac.transpose(1,2)@z/nn;cx=vc.transpose(1,2)@z/nn;slopes['X'][0,h]=bx;slopes['X'][1,h]=cx
  covA=ac.transpose(1,2)@ac/nn;ua,ra,ia,ga=eigroot(covA,64,3);bk=ua*ra[:,None];ck=(vc.transpose(1,2)@ac/nn)@(ua*ia[:,None]);slopes['K'][0,h]=bk;slopes['K'][1,h]=ck;keyranks.append(ga.sum(1).cpu())
  globalA=((a[h].double()-a[h].double().mean(0))**2).sum();globalV=((v[h].double()-v[h].double().mean(0))**2).sum();withinA=(ac**2).sum();withinV=(vc**2).sum()
  for m,b,c in [('X',bx,cx),('K',bk,ck)]:
   resA=withinA-(b.square().sum((1,2))*n).sum();resV=withinV-(c.square().sum((1,2))*n).sum();explained[m].append(dict(key_total_SSE=float(globalA),value_total_SSE=float(globalV),key_within_SSE=float(withinA),value_within_SSE=float(withinV),key_residual_SSE=float(resA),value_residual_SSE=float(resV)))
  am=a[h,jx].double();vm=v[h,jx].double();mam=(am*jw[...,None]).sum(1)/jn[:,None];mvm=(vm*jw[...,None]).sum(1)/jn[:,None];meansM[0,h]=mam;meansM[1,h]=mvm;resA=(((am-mam[:,None])*jw[...,None])**2).sum();resV=(((vm-mvm[:,None])*jw[...,None])**2).sum();explained['M'].append(dict(key_total_SSE=float(globalA),value_total_SSE=float(globalV),key_residual_SSE=float(resA),value_residual_SSE=float(resV)))
 states={m:dict(means=means,slopes=slopes[m],counts=n,nodes=nodes) for m in ('X','K')};states['M']=dict(means=meansM,slopes=None,counts=jn,nodes=jnodes);sync();stats=dict(seconds=time.monotonic()-begin,source_tokens=a.shape[1],coarse_groups=G,fine_groups=J,geometry_rank=good.sum(1).cpu(),geometry_mean=mu.cpu(),geometry_covariance=cov.cpu(),key_rank=torch.stack(keyranks),explained=explained,source_precision='FP64 common;encodingFP32',group_source_indices={str(k):torch.from_numpy(v.copy()).to(torch.int32) for k,v in coarse.items()},fine_group_source_indices={str(k):torch.from_numpy(v.copy()).to(torch.int32) for k,v in fine.items()})
 return states,stats

def encode(path,state,method):
 """Exact fixed format, no pickle overhead or retained geometry covariance."""
 t=time.monotonic();G=len(state['counts']);s=state['slopes'];header=struct.pack('<8s7q',b'SRMM001\0',G,16,64,0 if s is None else 3,1,0,0);assert len(header)==64;path.parent.mkdir(parents=True,exist_ok=True)
 nodes=np.asarray(state['nodes'],dtype='<i8');counts=state['counts'].cpu().numpy().astype('<i8');means=state['means'].cpu().numpy().astype('<f4');slopes=None if s is None else s.cpu().numpy().astype('<f4');tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:
  f.write(header)
  for g in range(G):
   f.write(nodes[g].tobytes());f.write(counts[g].tobytes());f.write(means[:,:,g].tobytes())
   if slopes is not None:f.write(slopes[:,:,g].tobytes())
 tmp.replace(path);expected=64+G*(16+d.MEAN+(0 if s is None else d.SLOPE));assert path.stat().st_size==expected
 return dict(method=method,path=str(path),groups=G,bytes=expected,sha256=d.sha(path),write_CPU_transfer_hash_seconds=time.monotonic()-t)
def decode(path,device):
 t=time.monotonic()
 with path.open('rb') as f:
  magic,G,H,D,R,*_=struct.unpack('<8s7q',f.read(64));assert magic==b'SRMM001\0' and H==16 and D==64
  nodes=np.empty(G,np.int64);counts=np.empty(G,np.int64);means=np.empty((2,H,G,D),np.float32);slopes=np.empty((2,H,G,D,3),np.float32) if R else None
  for g in range(G):
   nodes[g]=np.frombuffer(f.read(8),'<i8')[0];counts[g]=np.frombuffer(f.read(8),'<i8')[0];means[:,:,g]=np.frombuffer(f.read(d.MEAN),'<f4').reshape(2,H,D)
   if R:slopes[:,:,g]=np.frombuffer(f.read(d.SLOPE),'<f4').reshape(2,H,D,3)
  assert not f.read(1)
 state=dict(means=torch.from_numpy(means).to(device),slopes=None if slopes is None else torch.from_numpy(slopes).to(device),counts=torch.from_numpy(counts).to(device),nodes=nodes);sync() if str(device).startswith('cuda') else None
 return state,time.monotonic()-t

def approximation(r,state,logD,oD):
 ma,mv=state['means'];n=state['counts'];logits=torch.einsum('hqd,hgd->hqg',r,ma)+n.float().log()[None,None];slopes=state['slopes']
 if slopes is not None:
  b,c=slopes;s=torch.einsum('hqd,hgdr->hqgr',r,b);logits=logits+.5*s.square().sum(-1)
 logR=torch.logsumexp(logits,-1);logT=torch.logaddexp(logD,logR);weights=(logits-logT[...,None]).exp();out=(logD-logT).exp()[...,None]*oD+weights@mv
 if slopes is not None:out=out+torch.einsum('hqgr,hgdr->hqd',weights[...,None]*s,c)
 return out,logT,(logR-logT).exp()

def responses(q,a,v,native,weight,remote,states):
 begin=time.monotonic();r=q.float()/8.;a=a.float();v=v.float();D=torch.ones(a.shape[1],dtype=torch.bool,device=a.device);D[remote]=False;did=D.nonzero().flatten();vd=v[:,did];logs=[];projected=[];readseconds={m:0. for m in states};exactseconds=0.;validation=[]
 for first in range(0,1041,32):
  rr=r[:,first:first+32];s=time.monotonic();L=rr@a.transpose(1,2);logF=torch.logsumexp(L,-1);oF=(L-logF[...,None]).exp()@v;LD=L[:,:,did];logD=torch.logsumexp(LD,-1);oD=(LD-logD[...,None]).exp()@vd;logR=torch.logsumexp(L[:,:,remote],-1);massR=(logR-logF).exp();sync();exactseconds+=time.monotonic()-s;out=[oD];records=[logF,logD,logR,massR]
  for m in ('X','K','M'):
   s=time.monotonic();om,lz,mass=approximation(rr,states[m],logD,oD);sync();readseconds[m]+=time.monotonic()-s;out.append(om);records.extend([lz,mass])
  out.append(native[:,first:first+32].float());deltas=torch.stack([o-oF for o in out]).permute(0,2,1,3).reshape(5,-1,1024);pd=torch.nn.functional.linear(deltas,weight.float(),None);projected.append(pd.cpu());logs.append(torch.stack(records).cpu())
  if first==0:validation=[oF[:,:6].cpu(),*[x[:,:6].cpu() for x in out]]
 sync();return dict(projected=torch.cat(projected,1),logs=torch.cat(logs,2),projection_methods=['D','X','K','M','native_fused'],log_fields=['logF','logD','logR','massR','logX','massX','logK','massK','logM','massM'],remote_tokens=len(remote),exact_tokens=len(did),timing=dict(total_seconds=time.monotonic()-begin,exact_discrete_seconds=exactseconds,encoded_read_seconds=readseconds)),validation
