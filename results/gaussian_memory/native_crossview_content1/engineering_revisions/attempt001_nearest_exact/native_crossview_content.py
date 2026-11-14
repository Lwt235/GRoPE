"""Round025: geometry-only locked correspondence, then cached-vector arithmetic. No model."""
import os,sys,json,time,hashlib,datetime,fcntl,signal,resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_crossview_content1'
DATA=Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory/native_crossview_content_v1')
ROUND=ROOT/'results/gaussian_memory/autonomous_research_loop/round_025'
SCIENCE_SHA='0a9018d6843177bb122c399dd0dceb9e8587765c578dd615970b677de94c14af'
STOP=False
def stop(*a):
    global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    q.replace(p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');torch.save(x,q)
    with q.open('rb') as f:os.fsync(f.fileno())
    q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def log(name,kind,**kw):
    p=OUT/'calls'/f'{name}.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(utc=now(),kind=kind,**kw))+'\n');f.flush()
COLS=['grid_points','valid_anchor_points','common_points','coverage','depth_a','depth_b','scale_a_major','scale_a_minor','scale_b_major','scale_b_minor','centroid_a_x','centroid_a_y','centroid_b_x','centroid_b_y','world_x','world_y','world_z','depth_forward_mean_m','depth_forward_max_m','depth_reverse_mean_m','depth_reverse_max_m','reprojection_a_mean_px','reprojection_a_max_px','reprojection_b_mean_px','reprojection_b_max_px','view_angle_degrees']
def prepare():
    assert sha(ROUND/'science_plan.md')==SCIENCE_SHA
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (OUT/'configuration.json').exists():return
    old=read(ROOT/'results/gaussian_memory/native_content_support1/configuration.json');facts=read(ROUND/'crossview_input_facts.json');cases=[]
    for rid in sorted({x['id'] for x in old['images']}):
        images=[x for x in old['images'] if x['id']==rid];sources=[x for x in images if x['cohort']=='source'];queries=[x for x in images if x['cohort']=='query'];sourcepair=next(x['path'] for x in facts['original_source_pairs'] if f'/{rid}/' in x['path'])
        for a in queries:
            for b in sources:cases.append(dict(key=f'{rid}_q{a["frame"]}_s{b["frame"]}',id=rid,place=a['place'],cohort='query_source',a=a,b=b,source_pair=sourcepair))
        for i,a in enumerate(sources):
            for b in sources[i+1:]:cases.append(dict(key=f'{rid}_s{a["frame"]}_s{b["frame"]}',id=rid,place=a['place'],cohort='source_source',a=a,b=b,source_pair=sourcepair))
    assert len(cases)==264
    write(OUT/'configuration.json',dict(science_sha256=SCIENCE_SHA,cases=cases,images=old['images'],
        geometry=dict(grid=210,pixel_centers='edge coordinates x+.5,y+.5; K_metric query, K_source*210/448',source_resize='torch nearest-exact for depth_sensor and common_valid448→210; matches PIL nearest coordinate convention',depth_sampling='nearest210 center floor(projected edge coordinate), no interpolated invalid/fill depth',
        bilateral='Anchor point→b projection, sample nearest valid b center, backproject b center/depth→a, sample nearest valid a center/depth→b; forward/reverse measured-depth tolerance max(.02m,.02*measuredDepth), both cycle reprojection distances≤1px.',
        feature_points='Same original anchor world points projected continuously into both views; border bilinear at32grid with patch center(.5,.5) in patch units.',
        region='Original14x14patch footprint in448;210pixel center membership floor(32*p/210). Uniform anchor-grid point weights. Common>=16 and>=.5*all footprint grid points.',
        matching='Ascending original patchID; pair each unused eligible region with closest remaining unused eligible region, stableID ties; no replacement. Both-view mean-depth ratio<=1.25; both sorted projected RMS principal-axis scale ratios<=1.5; coverage difference<=.10; positive-weight interpolation index sets disjoint in EACH view.',
        distance='Mean squared normalized seven differences: two (max/min depth−1)/.25, four (max/min principal scale−1)/.5, abs(coverage difference)/.10; equal weight. 0/0 scale ratio=1, positive/0=infinity, no epsilon/tuning.',statistics_columns=COLS),
        feature=dict(channels=1024,original_full=True,normalization='None; no PCA/whitening/centering/norm matching',aggregation='FP64 uniform bilinear point sums; region mean rounded toFP32 once, saved; six squared distances and E inFP64 from those exact saved vectors',all_tables_locked_before_feature_values=True),
        limits=dict(GPU_worker_hours=1,new_DATA_GiB=4,new_project_GiB=1),engineering_reserves=dict(host_RAM_GiB=32,per_worker_GiB=3,manager_GiB=2,no_fixed_card_count=True),training=0,model_forward=0,predictions=0,gradients=0,renders=0,downloads=0,all_data_used=True))
    print(dict(prepared_pairs=264,query_source=192,source_source=72),flush=True)

def view_geometries(cases,name):
    source=load(cases[0]['source_pair']);ds=F.interpolate(source['depth_sensor'][:,None],size=(210,210),mode='nearest-exact')[:,0];vs=F.interpolate(source['common_valid'][:,None].float(),size=(210,210),mode='nearest-exact')[:,0].bool()
    result={}
    for image in [x for x in read(OUT/'configuration.json')['images'] if x['id']==cases[0]['id']]:
        if image['cohort']=='source':
            ix=image['source_index'];K=source['K'][ix].double().clone();K[:2]*=210/448;d=ds[ix];v=vs[ix];cam=source['c2w'][ix]
        else:
            q=load(image['image_path']);d=q['target'];v=q['valid'];K=q['K_metric'].double();cam=q['c2w']
        assert d.shape==v.shape==(210,210) and torch.isfinite(d[v]).all() and (d[v]>0).all()
        result[image['key']]=dict(depth=d.cuda().double(),valid=v.cuda(),K=K.cuda(),cam=cam.cuda().double())
    log(name,'geometry_inputs',scene=cases[0]['id'],views=8,source_sensor_mask_preserved=True,features_read=False)
    return result
def project(world,g):
    c=(world-g['cam'][:3,3])@g['cam'][:3,:3];p=c@g['K'].T;z=c[:,2];return p[:,:2]/z[:,None],z
def unproject(uv,depth,g):
    pixels=torch.cat((uv,torch.ones_like(uv[:,:1])),-1);camera=(pixels@torch.linalg.inv(g['K']).T)*depth[:,None]
    return camera@g['cam'][:3,:3].T+g['cam'][:3,3]
def sample(uv,g):
    inside=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<210).all(-1);index=torch.nan_to_num(uv,nan=0,posinf=209,neginf=0).floor().long().clamp(0,209);ix=index[:,1]*210+index[:,0]
    return g['depth'].reshape(-1)[ix],g['valid'].reshape(-1)[ix]&inside,index.double()+.5,inside
def aggregate(ids,values,n=1024):
    out=torch.zeros((n,values.shape[1]),device=values.device,dtype=values.dtype);out.index_add_(0,ids,values);return out
def support(ids,uv):
    xy=(uv*32/210-.5).clamp(0,31);lo=xy.floor().long();hi=(lo+1).clamp_max(31);f=xy-lo
    mask=torch.zeros(1024*1024,dtype=torch.bool,device=uv.device)
    for ax,ay,w in [(lo[:,0],lo[:,1],(1-f[:,0])*(1-f[:,1])),(hi[:,0],lo[:,1],f[:,0]*(1-f[:,1])),(lo[:,0],hi[:,1],(1-f[:,0])*f[:,1]),(hi[:,0],hi[:,1],f[:,0]*f[:,1])]:
        positive=w>0;mask[ids[positive]*1024+ay[positive]*32+ax[positive]]=True
    return mask.reshape(1024,1024)
def ratio_matrix(v):
    lo=torch.minimum(v[:,None],v[None,:]);hi=torch.maximum(v[:,None],v[None,:]);return torch.where(lo>0,hi/lo,torch.where(hi==0,1.,torch.inf))
@torch.inference_mode()
def geometry(case,views,name):
    marker=OUT/'geometry'/f'{case["key"]}.json'
    if marker.exists():return
    begun=time.monotonic();a,b=views[case['a']['key']],views[case['b']['key']];device=a['depth'].device
    yy,xx=torch.meshgrid(torch.arange(210,device=device,dtype=torch.float64)+.5,torch.arange(210,device=device,dtype=torch.float64)+.5,indexing='ij');pa=torch.stack((xx,yy),-1).reshape(-1,2);reg=(pa[:,1]*32/210).long()*32+(pa[:,0]*32/210).long();da=a['depth'].reshape(-1);anchorvalid=a['valid'].reshape(-1)
    world=unproject(pa,da,a);pb,zb=project(world,b);db,vb,qb,inb=sample(pb,b);wb=unproject(qb,db,b);pba,zba=project(wb,a);dar,va,qa,ina=sample(pba,a);war=unproject(qa,dar,a);pbr,zbr=project(war,b)
    fd=(zb-db).abs();rd=(zba-dar).abs();ra=(pba-pa).norm(dim=-1);rb=(pbr-qb).norm(dim=-1)
    stages=[('anchor_valid',anchorvalid),('positive_target_depth',zb>0),('target_in_image',inb),('target_valid',vb),('forward_depth',fd<=torch.maximum(db*.02,db.new_tensor(.02))),('reverse_in_image',ina),('reverse_valid',va),('reverse_depth',rd<=torch.maximum(dar*.02,dar.new_tensor(.02))),('reprojection_a',ra<=1),('cycle_target_front_and_in_image',(zbr>0)&torch.isfinite(pbr).all(-1)&(pbr>=0).all(-1)&(pbr<210).all(-1)),('reprojection_b',rb<=1)]
    passed=torch.ones_like(anchorvalid);counts=[]
    for label,ok in stages:
        prior=int(passed.sum());passed&=ok;counts.append(dict(stage=label,remaining=int(passed.sum()),newly_rejected=prior-int(passed.sum())))
    ix=torch.where(passed)[0];ids=reg[ix];count=torch.bincount(ids,minlength=1024).double();gridcount=torch.bincount(reg,minlength=1024).double();validcount=torch.bincount(reg[anchorvalid],minlength=1024).double();covg=count/gridcount;validregion=(count>=16)&(covg>=.5);den=count.clamp_min(1)
    def avg(v):return aggregate(ids,v)/den[:,None]
    paa,pbb=pa[ix],pb[ix];wa=world[ix];ma=avg(paa);mb=avg(pbb);mw=avg(wa)
    def axes(v,mu):
        diff=v-mu[ids];cov=aggregate(ids,(diff[:,:,None]*diff[:,None,:]).reshape(-1,4))/den[:,None]
        return torch.linalg.eigvalsh(cov.reshape(-1,2,2)).clamp_min(0).sqrt().flip(-1)
    sa,sb=axes(paa,ma),axes(pbb,mb);deptha=avg(da[ix,None])[:,0];depthb=avg(zb[ix,None])[:,0]
    raysa=a['cam'][:3,3]-mw;raysb=b['cam'][:3,3]-mw;angle=torch.acos(F.cosine_similarity(raysa,raysb,dim=-1).clamp(-1,1))*180/np.pi
    residual=torch.stack((fd[ix],rd[ix],ra[ix],rb[ix]),-1);rmean=avg(residual);rmax=torch.zeros(1024,4,device=device,dtype=torch.float64)
    if len(ix):rmax.scatter_reduce_(0,ids[:,None].expand(-1,4),residual,reduce='amax',include_self=True)
    stats=torch.stack((gridcount,validcount,count,covg,deptha,depthb,sa[:,0],sa[:,1],sb[:,0],sb[:,1],ma[:,0],ma[:,1],mb[:,0],mb[:,1],mw[:,0],mw[:,1],mw[:,2],rmean[:,0],rmax[:,0],rmean[:,1],rmax[:,1],rmean[:,2],rmax[:,2],rmean[:,3],rmax[:,3],angle),-1)
    spa,spb=support(ids,paa),support(ids,pbb);eligible=validregion[:,None]&validregion[None,:];dist=torch.zeros(1024,1024,device=device,dtype=torch.float64)
    for v,tol in [(deptha,.25),(depthb,.25),(sa[:,0],.5),(sa[:,1],.5),(sb[:,0],.5),(sb[:,1],.5)]:
        ratio=ratio_matrix(v);eligible&=ratio<=1+tol;dist+=((ratio-1)/tol).square()
    cd=(covg[:,None]-covg[None,:]).abs();eligible&=cd<=.10;dist+=(cd/.10).square();dist/=7
    # Binary integer overlaps are exactly representable with TF32 disabled.
    overlap_a=spa.float()@spa.float().T;overlap_b=spb.float()@spb.float().T;eligible&=(overlap_a==0)&(overlap_b==0);eligible.fill_diagonal_(False)
    D=torch.where(eligible,dist,torch.inf).cpu().numpy();qualified=validregion.cpu().numpy();used=np.zeros(1024,dtype=bool);pairs=[]
    for r in range(1024):
        if used[r] or not qualified[r]:continue
        row=D[r].copy();row[used]=np.inf;t=int(np.argmin(row))
        if not np.isfinite(row[t]):continue
        pairs.append([r,t]);used[[r,t]]=True
    matched=torch.tensor(pairs,dtype=torch.int64).reshape(-1,2);packed_a=np.packbits(spa.cpu().numpy(),axis=1);packed_b=np.packbits(spb.cpu().numpy(),axis=1)
    payload=dict(anchor_point_indices=ix.cpu().to(torch.int32),projected_b=pb[ix].cpu(),point_residuals=residual.cpu().float(),region_statistics=stats.cpu(),valid_regions=validregion.cpu(),matched_regions=matched,geometry_pair_distance=torch.tensor([D[r,t] for r,t in pairs],dtype=torch.float64),support_a_packed=torch.from_numpy(packed_a),support_b_packed=torch.from_numpy(packed_b),statistics_columns=COLS,stage_counts=counts,case=case)
    target=DATA/'geometry'/f'{case["key"]}.pt';save(target,payload);torch.cuda.synchronize()
    record=dict(complete=True,key=case['key'],case=case,data_path=str(target),data_bytes=target.stat().st_size,sha256=sha(target),candidate_regions=1024,qualified_regions=int(validregion.sum()),matched_region_pairs=len(pairs),matched_regions=int(used.sum()),qualified_unmatched_regions=int(validregion.sum())-int(used.sum()),common_world_points=len(ix),grid_world_points=44100,stage_counts=counts,
        region_reasons=dict(fewer_than16_common=int((count<16).sum()),coverage_below_half=int((covg<.5).sum()),qualified=int(validregion.sum()),qualified_unmatched=int(validregion.sum())-int(used.sum())),feature_values_read=False,elapsed_seconds=time.monotonic()-begun,ended_utc=now())
    write(marker,record);log(name,'geometry_pair_complete',key=case['key'],seconds=record['elapsed_seconds'],regions=int(validregion.sum()),matched_pairs=len(pairs));print(dict(geometry=case['key'],matches=len(pairs)),flush=True)

def features(cases,name):
    freeze=read(OUT/'geometry_lock.json');assert freeze['complete'] and freeze['pairs']==264
    cache={};sourcefile=cases[0]['b']['cached_features'];t=time.monotonic();src=load(sourcefile)['features'];log(name,'cached_feature_file_read',path=sourcefile,scene=cases[0]['id'],after_geometry_lock_sha=sha(OUT/'geometry_lock.json'))
    for x in [v for v in read(OUT/'configuration.json')['images'] if v['id']==cases[0]['id']]:
        if x['cohort']=='source':f=src[x['source_index']].reshape(1024,1024)
        else:
            f=load(x['cached_features'])['patches'].reshape(1024,1024);log(name,'cached_feature_file_read',path=x['cached_features'],scene=x['id'],after_geometry_lock_sha=sha(OUT/'geometry_lock.json'))
        assert torch.isfinite(f).all();cache[x['key']]=f.cuda().float()
    torch.cuda.synchronize();log(name,'feature_materialization',scene=cases[0]['id'],seconds=time.monotonic()-t,views=8,channels=1024)
    return cache

def bilinear(f,uv):
    xy=(uv*32/210-.5).clamp(0,31);lo=xy.floor().long();hi=(lo+1).clamp_max(31);frac=xy-lo
    value=torch.zeros(len(uv),1024,device=f.device,dtype=torch.float64)
    for xx,yy,w in [(lo[:,0],lo[:,1],(1-frac[:,0])*(1-frac[:,1])),(hi[:,0],lo[:,1],frac[:,0]*(1-frac[:,1])),(lo[:,0],hi[:,1],(1-frac[:,0])*frac[:,1]),(hi[:,0],hi[:,1],frac[:,0]*frac[:,1])]:value+=f[yy*32+xx].double()*w[:,None]
    return value
@torch.inference_mode()
def measure(case,cache,name):
    marker=OUT/'measurements'/f'{case["key"]}.json'
    if marker.exists():return
    begin=time.monotonic();gm=read(OUT/'geometry'/f'{case["key"]}.json');g=load(gm['data_path']);pairs=g['matched_regions'].cuda();n=len(pairs);device=cache[case['a']['key']].device
    used=torch.unique(pairs.flatten(),sorted=True);regions_used=len(used);inverse=torch.full((1024,),-1,device=device,dtype=torch.long);inverse[used]=torch.arange(regions_used,device=device)
    points=g['anchor_point_indices'].cuda().long();pa=torch.stack(((points%210).double()+.5,(points//210).double()+.5),-1);rid=(pa[:,1]*32/210).long()*32+(pa[:,0]*32/210).long();take=inverse[rid]>=0;pa=pa[take];pb=g['projected_b'].cuda()[take];rids=inverse[rid[take]]
    accum=torch.zeros(regions_used,2,1024,device=device,dtype=torch.float64);cnt=torch.bincount(rids,minlength=regions_used).double()
    for first in range(0,len(pa),512):
        ids=rids[first:first+512]
        for vi,(key,uv) in enumerate(((case['a']['key'],pa),(case['b']['key'],pb))):accum[:,vi].index_add_(0,ids,bilinear(cache[key],uv[first:first+512]))
    means=(accum/cnt.clamp_min(1)[:,None,None]).float();v=means.double();pidx=inverse[pairs];x=v[pidx[:,0],0];y=v[pidx[:,0],1];u=v[pidx[:,1],0];z=v[pidx[:,1],1]
    def d(a,b):return (a-b).square().mean(-1)
    es=(d(x,y)+d(u,z))/2;sw=(d(x,z)+d(u,y))/2;wi=(d(x,u)+d(y,z))/2
    assert torch.isfinite(means).all() and torch.isfinite(es).all() and torch.isfinite(sw).all() and torch.isfinite(wi).all()
    if n:assert ((sw-es).abs()<=wi+1e-12).all()
    target=DATA/'vectors'/f'{case["key"]}.pt';save(target,dict(region_ids=used.cpu(),vectors=means.cpu(),matched_regions=g['matched_regions'],E_same=es.cpu(),E_swap=sw.cpu(),E_within=wi.cpu(),geometry_sha256=gm['sha256'],case=case))
    torch.cuda.synchronize();s=float(es.mean()) if n else None;t=float(sw.mean()) if n else None;w=float(wi.mean()) if n else None
    record=dict(complete=True,key=case['key'],case=case,data_path=str(target),data_bytes=target.stat().st_size,sha256=sha(target),geometry_sha256=gm['sha256'],region_pairs=n,regions=regions_used,world_points_averaged=len(pa),E_same=s,E_swap=t,E_within=w,K=(t-s)/w if w is not None and w>0 else None,R=s/w if w is not None and w>0 else None,zero_within_region_pairs=int((wi==0).sum()),undefined_reason='no_geometry_matched_region_pairs' if not n else ('zero_spatial_denominator' if w==0 else None),elapsed_seconds=time.monotonic()-begin,ended_utc=now())
    write(marker,record);log(name,'vector_pair_complete',key=case['key'],seconds=record['elapsed_seconds'],region_pairs=n,distances=6*n,region_vectors=2*regions_used,world_point_view_interpolations=2*len(pa));print(dict(measurement=case['key'],region_pairs=n,K=record['K'],R=record['R']),flush=True)

def worker(phase,rid,name):
    torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;signal.signal(signal.SIGTERM,stop);OUT.mkdir(exist_ok=True);torch.cuda.init();torch.cuda.synchronize();log(name,'CUDA_access',device=torch.cuda.get_device_name(0),phase=phase)
    cases=[x for x in read(OUT/'configuration.json')['cases'] if x['id']==rid];cache=view_geometries(cases,name) if phase=='geometry' else features(cases,name)
    for case in cases:
        if STOP:break
        marker=OUT/('geometry' if phase=='geometry' else 'measurements')/f'{case["key"]}.json'
        if marker.exists():continue
        lock=marker.with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if phase=='geometry':geometry(case,cache,name)
            else:measure(case,cache,name)
    write(OUT/'worker_resources'/f'{name}.json',dict(peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_used=True,phase=phase,scene=rid,ended_utc=now(),terminated_after_complete_pair=STOP))
    if STOP:raise SystemExit(75)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','geometry','features']);p.add_argument('--id');p.add_argument('--name');a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:worker(a.phase,a.id,a.name)
