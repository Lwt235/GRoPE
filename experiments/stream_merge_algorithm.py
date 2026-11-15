"""All-dimensional Ward candidates and source-only pooled spatial moments."""
import torch,time

def geometry(depth,pose):
 from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
 ext,K=pose_encoding_to_extri_intri(pose[None,None],(392,518));ext=ext[0,0].double();K=K[0,0].double();yy,xx=torch.meshgrid(torch.arange(392,device=depth.device),torch.arange(518,device=depth.device),indexing='ij');uv=torch.stack((xx+.5,yy+.5,torch.ones_like(xx)),-1).double();z=depth.double();local=(uv@torch.linalg.inv(K).T)*z[...,None];world=(local-ext[:,3])@ext[:,:3];points=world.reshape(28,14,37,14,3).permute(0,2,1,3,4).reshape(1036,196,3);zz=z.reshape(28,14,37,14).permute(0,2,1,3).reshape(1036,196);valid=torch.isfinite(points).all(-1)&torch.isfinite(zz)&(zz>0)&torch.isfinite(pose).all();count=valid.sum(1);m=(count>0).long();p=torch.where(valid[...,None],points,torch.zeros_like(points));mu=p.sum(1)/count.clamp_min(1)[:,None];center=torch.where(valid[...,None],points-mu[:,None],torch.zeros_like(points));cov=center.transpose(1,2)@center/count.clamp_min(1)[:,None,None];sortedz=zz.masked_fill(~valid,float('inf')).sort(1).values;median=(sortedz.gather(1,((count-1).clamp_min(0)//2)[:,None])[:,0]+sortedz.gather(1,(count//2)[:,None])[:,0])*.5;scale=median.square()*(K[0,0].reciprocal().square()+K[1,1].reciprocal().square())/2;good=torch.isfinite(scale)&(scale>0)&(m>0);vv=scale[good].sort().values;delta=float((vv[(len(vv)-1)//2]+vv[len(vv)//2])*.5) if len(vv) else None;return mu,cov,m,delta

def pooled(mu,cov,m,i,j):
 mi=m[i].double();mj=m[j].double();total=mi+mj;wi=mi/total.clamp_min(1);wj=mj/total.clamp_min(1);diff=mu[i]-mu[j];mean=wi[...,None]*mu[i]+wj[...,None]*mu[j];S=wi[...,None,None]*cov[i]+wj[...,None,None]*cov[j]+(wi*wj)[...,None,None]*diff[...,None]*diff[...,None,:];return mean,S,total

def gscore(policy,mu,cov,m,i,j,delta):
 mean,S,total=pooled(mu,cov,m,i,j)
 if policy=='I':
  si=cov[i].diagonal(dim1=-2,dim2=-1).sum(-1)/3;sj=cov[j].diagonal(dim1=-2,dim2=-1).sum(-1)/3;ss=S.diagonal(dim1=-2,dim2=-1).sum(-1)/3;return 3*(total*(ss+delta).log()-m[i]*(si+delta).log()-m[j]*(sj+delta).log())
 eye=torch.eye(3,device=mu.device,dtype=torch.float64)*delta;sign,ld=torch.linalg.slogdet(S+eye);si,li=torch.linalg.slogdet(cov[i]+eye);sj,lj=torch.linalg.slogdet(cov[j]+eye);assert bool(((sign>0)&(si>0)&(sj>0)).all()),'Nonpositive regularized support matrix';return total*ld-m[i]*li-m[j]*lj

def content_matrix(f,n):
 # FP64 norms in bounded dimension chunks, no hidden full FP64 content copy.
 norms=torch.zeros(len(f),device=f.device,dtype=torch.float64)
 for k in range(0,f.shape[1],1024):
  block=f[:,k:k+1024].double();norms+=block.square().sum(1)
 dot=f@f.T;D=dot.double();del dot;D.mul_(-2).add_(norms[:,None]).add_(norms[None,:]).clamp_min_(0);nn=n.double();D.mul_(nn[:,None]).mul_(nn[None,:]).div_(nn[:,None]+nn[None,:]);r=torch.arange(len(f),device=f.device);D.masked_fill_(r[:,None]>=r[None,:],float('inf'));return D,norms

def choose(C,active,m,mu,cov,policy,delta,N):
 L=len(C);invalid=active&(m==0)
 if bool(invalid.any()):
  allowed=invalid[:,None]|invalid[None,:];flat=C.masked_fill(~allowed,float('inf')).flatten();idx=int(flat.argmin());return idx//L,idx%L,torch.zeros((),device=C.device,dtype=torch.float64),'missing_geometry_content'
 if policy=='C' or delta is None:
  idx=int(C.flatten().argmin());return idx//L,idx%L,torch.zeros((),device=C.device,dtype=torch.float64),'C' if policy=='C' else 'scale_undefined_content'
 k=min(N,N*(N-1)//2);values,indices=torch.topk(C.flatten(),k,largest=False,sorted=False);cut=values.max();strict=indices[values<cut];need=k-len(strict);ties=(C.flatten()==cut).nonzero().flatten()[:need];idx=torch.cat((strict,ties));i=idx//L;j=idx%L;score=gscore(policy,mu,cov,m,i,j,delta);minimum=score.min();best=int(torch.where(score==minimum,idx,torch.full_like(idx,2**62)).min());which=(idx==best).nonzero()[0,0];return best//L,best%L,score[which],policy

def merge(state,incoming,geo,IDs,policy,M):
 begin=time.monotonic();means=torch.cat((state['means'],incoming.float()),0);n=torch.cat((state['n'],torch.ones(len(incoming),device='cuda',dtype=torch.int64)));m=torch.cat((state['m'],geo[2]));mu=torch.cat((state['mu'],geo[0]));cov=torch.cat((state['cov'],geo[1]));ids=torch.cat((state['remote_ids'],IDs));N=len(n);initial=N;events=[]
 if N<=M:
  state.update(means=means,n=n,m=m,mu=mu,cov=cov,remote_ids=ids);return dict(seconds=time.monotonic()-begin,merges=0,entered=len(incoming),slots=N,mass=int(n.sum()),missing_geometry_slots=int((m==0).sum()),events=[],costs=torch.empty((0,2)))
 scale=state['scale'][...,None].expand(24,2,16,64).reshape(-1);f=means.reshape(N,-1)/scale;C,norms=content_matrix(f,n);active=torch.ones(N,device='cuda',dtype=torch.bool);nl=n.cpu().tolist();ml=m.cpu().tolist();il=ids.cpu().tolist();costs=torch.empty((N-M,2),device='cuda',dtype=torch.float64);counts={}
 for step in range(N-M):
  i,j,g,why=choose(C,active,m,mu,cov,policy,state['delta'],N-step);assert i<j;ni=nl[i];nj=nl[j];mi=ml[i];mj=ml[j];total=ni+nj;costs[step,0]=C[i,j];costs[step,1]=g;events.append([il[i],il[j],ni,nj,mi,mj]);counts[why]=counts.get(why,0)+1
  pm,ps,mt=pooled(mu,cov,m,i,j);mu[i]=pm;cov[i]=ps;m[i]=mi+mj;means[i].mul_(ni/total).add_(means[j],alpha=nj/total);n[i]=total;n[j]=0;m[j]=0;nl[i]=total;nl[j]=0;ml[i]=mi+mj;ml[j]=0;active[j]=False
  # Recompute exact current stored-FP32 centroid row, not recurrence on old distances.
  f[i]=means[i].reshape(-1)/scale;norms[i]=f[i].double().square().sum();row=(norms+norms[i]-2*(f@f[i]).double()).clamp_min_(0);row*=n.double()*total/(n.double()+total);row.masked_fill_(~active,float('inf'));row[i]=float('inf');C[:i,i]=row[:i];C[i,i+1:]=row[i+1:];C[j,:]=float('inf');C[:,j]=float('inf')
 keep=active.nonzero().flatten();del f,C,norms;state.update(means=means[keep],n=n[keep],m=m[keep],mu=mu[keep],cov=cov[keep],remote_ids=ids[keep]);cc=costs.cpu();return dict(seconds=time.monotonic()-begin,merges=initial-M,entered=len(incoming),slots=M,mass=int(state['n'].sum()),missing_geometry_slots=int((state['m']==0).sum()),events=events,costs=cc,selection_counts=counts,mass_quantiles=torch.quantile(state['n'].double(),torch.tensor([0,.25,.5,.75,1.],device='cuda',dtype=torch.float64)).cpu().tolist())
