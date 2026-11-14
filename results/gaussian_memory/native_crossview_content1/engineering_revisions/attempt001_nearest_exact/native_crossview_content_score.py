"""CPU-only independent saved-vector scoring, coverage and geometry/interface validation."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ['MPLCONFIGDIR']='/tmp/liwt_native_crossview_mpl'
import json,time,math,collections,resource,csv
import numpy as np
import torch
from PIL import Image
from experiments import native_crossview_content as d

def finite(x):
    if isinstance(x,dict):return {str(k):finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [finite(v) for v in x]
    if isinstance(x,np.ndarray):return finite(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    return x
def average(rows):
    valid=[x for x in rows if x['E_same'] is not None]
    if not valid:return dict(E_same=None,E_swap=None,E_within=None,K=None,R=None,contributing=len(valid))
    e={k:math.fsum(x[k] for x in valid)/len(valid) for k in ('E_same','E_swap','E_within')};w=e['E_within']
    return dict(**e,K=(e['E_swap']-e['E_same'])/w if w>0 else None,R=e['E_same']/w if w>0 else None,contributing=len(valid))
def bootstrap(values):
    values=np.array([v for v in values if v is not None],dtype=np.float64);n=len(values)
    if not n:return dict(mean=None,interval95=None,places=0)
    boot=values[np.random.default_rng(20261114).integers(0,n,(5000,n))].mean(1) if n>1 else None
    return dict(mean=float(values.mean()),interval95=np.percentile(boot,[2.5,97.5]).tolist() if boot is not None else None,places=n)

def score():
    start=time.monotonic();torch.set_num_threads(2);cfg=d.read(d.OUT/'configuration.json');lock=d.read(d.OUT/'geometry_lock.json');assert lock['complete'];pool=d.read(d.OUT/'pool_complete.json');assert pool['complete'] and pool['all_children_actual_wait']
    calls=[json.loads(l) for p in (d.OUT/'calls').glob('*.jsonl') for l in p.read_text().splitlines()];featureevents=[e for e in calls if e['kind']=='cached_feature_file_read'];assert all(e['utc']>lock['locked_utc'] and e['after_geometry_lock_sha']==d.sha(d.OUT/'geometry_lock.json') for e in featureevents)
    pairs=[];quads=[];geometries=[];max_e=0.;max_identity=0.;worst_match=0.;verify_count=0;zeroquad=0
    for number,case in enumerate(cfg['cases']):
        gm=d.read(d.OUT/'geometry'/f'{case["key"]}.json');mm=d.read(d.OUT/'measurements'/f'{case["key"]}.json');assert gm['sha256']==lock['geometry_data_identities'][case['key']]==mm['geometry_sha256'];assert d.sha(gm['data_path'])==gm['sha256'] and d.sha(mm['data_path'])==mm['sha256']
        g=d.load(gm['data_path']);v=d.load(mm['data_path']);regions=v['region_ids'].numpy();vectors=v['vectors'].numpy().astype(np.float64);matches=v['matched_regions'].numpy();st=g['region_statistics'].numpy();mapid={int(r):i for i,r in enumerate(regions)}
        assert len(np.unique(matches))==matches.size and len(matches)==gm['matched_region_pairs']
        aa=g['support_a_packed'].numpy();bb=g['support_b_packed'].numpy();accepted=g['valid_regions'].numpy();vals=[]
        for qi,(r,t) in enumerate(matches):
            r,t=int(r),int(t);assert accepted[r] and accepted[t];assert not np.bitwise_and(aa[r],aa[t]).any() and not np.bitwise_and(bb[r],bb[t]).any();a,b=st[r],st[t]
            normalized=[]
            for ix,tol in [(4,.25),(5,.25),(6,.5),(7,.5),(8,.5),(9,.5)]:
                lo=min(a[ix],b[ix]);hi=max(a[ix],b[ix]);ratio=hi/lo if lo>0 else (1 if hi==0 else np.inf);assert ratio<=1+tol+1e-12;normalized.append((ratio-1)/tol)
            assert abs(a[3]-b[3])<=.10+1e-12;normalized.append(abs(a[3]-b[3])/.10);dist=math.fsum(z*z for z in normalized)/7;worst_match=max(worst_match,abs(dist-float(g['geometry_pair_distance'][qi])))
            x,y=vectors[mapid[r]];u,z=vectors[mapid[t]]
            def dd(a,b):return float(np.dot(a-b,a-b)/1024)
            es=.5*(dd(x,y)+dd(u,z));sw=.5*(dd(x,z)+dd(u,y));wi=.5*(dd(x,u)+dd(y,z));diff=float(np.dot(x-u,y-z)/1024);max_identity=max(max_identity,abs(sw-es-diff));verify_count+=1
            for name,value in [('E_same',es),('E_swap',sw),('E_within',wi)]:max_e=max(max_e,abs(value-float(v[name][qi])))
            if wi==0:zeroquad+=1
            vals.append(dict(E_same=es,E_swap=sw,E_within=wi))
            quads.append(dict(key=case['key'],id=case['id'],place=case['place'],cohort=case['cohort'],a_frame=case['a']['frame'],b_frame=case['b']['frame'],r=r,t=t,E_same=es,E_swap=sw,E_within=wi,K=(sw-es)/wi if wi>0 else None,R=es/wi if wi>0 else None,
                geometry_distance=dist,coverage_r=float(a[3]),coverage_t=float(b[3]),depth_a_r=float(a[4]),depth_a_t=float(b[4]),depth_b_r=float(a[5]),depth_b_t=float(b[5]),
                scales_a_r=a[6:8].tolist(),scales_a_t=b[6:8].tolist(),scales_b_r=a[8:10].tolist(),scales_b_t=b[8:10].tolist(),view_angle_r_deg=float(a[25]),view_angle_t_deg=float(b[25]),
                separation_a_px=float(np.linalg.norm(a[10:12]-b[10:12])),separation_b_px=float(np.linalg.norm(a[12:14]-b[12:14])),separation_world_m=float(np.linalg.norm(a[14:17]-b[14:17])),
                forward_depth_max_m=float(max(a[18],b[18])),reverse_depth_max_m=float(max(a[20],b[20])),reprojection_a_max_px=float(max(a[22],b[22])),reprojection_b_max_px=float(max(a[24],b[24])),
                point_count_r=int(a[2]),point_count_t=int(b[2])))
        stat=average(vals);expected={k:mm[k] for k in ('E_same','E_swap','E_within','K','R')}
        for k in expected:
            if expected[k] is not None:max_e=max(max_e,abs(expected[k]-stat[k]))
            else:assert stat[k] is None
        allcommon=g['point_residuals'].numpy();used=st[regions] if len(regions) else np.empty((0,len(d.COLS)))
        qual=dict(stage_counts=gm['stage_counts'],common_points=gm['common_world_points'],qualified_regions=gm['qualified_regions'],matched_regions=gm['matched_regions'],unmatched_qualified=gm['qualified_unmatched_regions'],region_reasons=gm['region_reasons'],
            matched_region_summary={d.COLS[i]:np.percentile(used[:,i],[0,25,50,75,95,100]).tolist() if len(used) else None for i in [2,3,4,5,6,7,8,9,25]},
            all_common_residual_quantiles={k:np.percentile(allcommon[:,i],[0,50,95,100]).tolist() if len(allcommon) else None for i,k in enumerate(['forward_depth_m','reverse_depth_m','reprojection_a_px','reprojection_b_px'])})
        pairs.append(dict(case=case,**stat,region_pairs=len(vals),zero_within_region_pairs=mm['zero_within_region_pairs'],undefined_reason=mm['undefined_reason'],geometry=qual))
        geometries.append(dict(key=case['key'],cohort=case['cohort'],**qual))
    assert max_e<1e-10 and max_identity<1e-10 and worst_match<1e-10
    queries=[];sources=[];places=[]
    for rid in sorted({c['id'] for c in cfg['cases']}):
        rr=[p for p in pairs if p['case']['id']==rid];qs=[p for p in rr if p['case']['cohort']=='query_source'];aux=[p for p in rr if p['case']['cohort']=='source_source']
        localq=[]
        for q in (2,5,8,11):
            these=[p for p in qs if p['case']['a']['frame']==q];r=dict(id=rid,place=rr[0]['case']['place'],query=q,view_pairs=4,view_pairs_defined=sum(p['K'] is not None for p in these),region_pairs=sum(p['region_pairs'] for p in these),**average(these));queries.append(r);localq.append(r)
        for s in (0,3,6,9):
            these=[p for p in qs if p['case']['b']['frame']==s];sources.append(dict(id=rid,place=rr[0]['case']['place'],source=s,query_source_view_pairs=4,view_pairs_defined=sum(p['K'] is not None for p in these),region_pairs=sum(p['region_pairs'] for p in these),**average(these)))
        places.append(dict(id=rid,place=rr[0]['case']['place'],cohort='query_source',**average(localq),view_pairs=16,view_pairs_defined=sum(p['K'] is not None for p in qs),queries_defined=sum(q['K'] is not None for q in localq),region_pairs=sum(p['region_pairs'] for p in qs)))
        places.append(dict(id=rid,place=rr[0]['case']['place'],cohort='source_source',**average(aux),view_pairs=6,view_pairs_defined=sum(p['K'] is not None for p in aux),region_pairs=sum(p['region_pairs'] for p in aux)))
    cohorts={}
    for cohort in ['query_source','source_source']:
        pp=[p for p in places if p['cohort']==cohort];vv=[p for p in pairs if p['case']['cohort']==cohort];gg=[g for g in geometries if g['cohort']==cohort];qq=[q for q in quads if q['cohort']==cohort]
        cohorts[cohort]=dict(metrics={k:bootstrap([p[k] for p in pp]) for k in ('K','R','E_same','E_swap','E_within')},
            coverage=dict(fixed_view_pairs=len(vv),defined_view_pairs=sum(v['K'] is not None for v in vv),no_matched_region_pairs=sum(v['region_pairs']==0 for v in vv),zero_view_pair_denominator=sum(v['region_pairs']>0 and v['E_within']==0 for v in vv),region_pairs=len(qq),zero_within_region_pairs=sum(q['E_within']==0 for q in qq),candidate_regions=1024*len(vv),qualified_regions=sum(g['qualified_regions'] for g in gg),matched_regions=sum(g['matched_regions'] for g in gg),qualified_unmatched_regions=sum(g['unmatched_qualified'] for g in gg),defined_places=sum(p['K'] is not None for p in pp),undefined_places=[p['id'] for p in pp if p['K'] is None],negative_region_K=sum(q['K'] is not None and q['K']<0 for q in qq),negative_viewpair_K=sum(v['K'] is not None and v['K']<0 for v in vv)),
            interpretation='Primary source-balanced then query-balanced E within each place, then place K/R, places equally.' if cohort=='query_source' else 'Auxiliary fixed six earlier-anchor unordered sourcepairs; valid view-pair E equally within place then place K/R. Missing pairs areNA, not replaced; not an interaction test.')
    primary=cohorts['query_source']['metrics'];k=primary['K'];r=primary['R'];supported=k['interval95'] is not None and k['interval95'][0]>0
    branch='K_not_supported_or_coverage_undefined'
    if supported:branch='regional_signal_direct_substitution_basis' if r['interval95'][1]<1 else ('regional_signal_substitution_cost_large' if r['interval95'][0]>1 else 'regional_signal_substitution_unresolved')
    summary=dict(complete=True,cohorts=cohorts,bootstrap=dict(replicates=5000,seed=20261114,unit='Actual defined places only. Pair E equal regions, source/query equal before place ratios; no region/viewpair pseudoreplication. Auxiliary valid six pairs equally, no interaction inference.'),criterion=dict(K_positive_lower95=supported,R_relation_to1=branch),all_data_used=True,PE_or_Native_or_fresh=False)
    d.write(d.OUT/'summary.json',finite(summary));d.write(d.OUT/'viewpair_measurements.json',finite(pairs));d.write(d.OUT/'query_measurements.json',finite(queries));d.write(d.OUT/'source_measurements.json',finite(sources));d.write(d.OUT/'place_measurements.json',finite(places));d.write(d.OUT/'region_pair_measurements.json',finite(quads));d.write(d.OUT/'geometry_quality.json',finite(geometries))
    with (d.OUT/'region_pair_measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(quads[0]));w.writeheader();w.writerows(quads)
    d.write(d.OUT/'independent_verification.json',dict(complete=True,new_saved_vectors_only=True,verified_region_pairs=verify_count,max_E_and_ratio_absolute_difference=max_e,max_exchange_dot_identity_difference=max_identity,max_geometry_distance_difference=worst_match,geometry_constraints_and_disjoint_support_all_verified=True,geometry_global_lock_precedes_all_feature_reads=True,feature_file_reads=len(featureevents),zero_spatial_contrast_region_pairs_retained=zeroquad,no_old_metrics_recomputed=True,model_calls=0,CUDA_initialized=torch.cuda.is_initialized()))
    # Fixed first pair/interface cases in each cohort, not selected by effect direction.
    interface_checks(cfg,quads)
    selected=[]
    primaryq=[q for q in quads if q['cohort']=='query_source' and q['K'] is not None]
    for key,choose in [('Fixed first matched query region',lambda:min(primaryq,key=lambda q:(q['key'],q['r'],q['t']))),('Smallest regional K',lambda:min(primaryq,key=lambda q:q['K'])),('Largest substitution R',lambda:max(primaryq,key=lambda q:q['R']))]:selected.append(dict(choose(),label=key))
    av=[q for q in quads if q['cohort']=='source_source' and q['K'] is not None]
    if av:selected.append(dict(min(av,key=lambda q:(q['key'],q['r'],q['t'])),label='Fixed first auxiliary source pair'))
    d.write(d.OUT/'figure_selection.json',finite(dict(selected=selected,selection='Fixed first primary correspondence, complete-primary minimumK and maximumR, fixed first auxiliary. No retuning/matching changes.')))
    figures(summary,places,selected,cfg)
    d.write(d.OUT/'CPU_scoring_runtime.json',dict(complete=True,body_wall_seconds=time.monotonic()-start,CPU_process_seconds=time.process_time(),peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_gradient_render_calls=0,CUDA_initialized=torch.cuda.is_initialized()))
    print(json.dumps(finite(summary),indent=2),flush=True)

def interface_checks(cfg,quads):
    checks=[];source_checks=[];cases=[]
    for cohort in ('query_source','source_source'):cases.append(next(c for c in cfg['cases'] if c['cohort']==cohort and any(q['key']==c['key'] for q in quads)))
    for c in cases:
        gm=d.read(d.OUT/'geometry'/f'{c["key"]}.json');g=d.load(gm['data_path']);m=d.load(d.read(d.OUT/'measurements'/f'{c["key"]}.json')['data_path']);r=int(g['matched_regions'][0,0]);idx=np.flatnonzero(m['region_ids'].numpy()==r)[0];pt=g['anchor_point_indices'].numpy().astype(np.int64);pa=np.stack(((pt%210)+.5,(pt//210)+.5),-1);rid=(pa[:,1]*32/210).astype(int)*32+(pa[:,0]*32/210).astype(int);use=rid==r;uvs=[pa[use],g['projected_b'].numpy()[use]];errors=[]
        for vi,image in enumerate((c['a'],c['b'])):
            raw=d.load(image['cached_features']);f=raw['patches'].reshape(32,32,1024) if image['cohort']=='query' else raw['features'][image['source_index']];f=f.permute(2,0,1)[None].float();grid=torch.from_numpy(uvs[vi]*2/210-1).float().reshape(1,-1,1,2)
            sampled=torch.nn.functional.grid_sample(f,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0].double().mean(1).numpy();errors.append(float(np.max(np.abs(sampled-m['vectors'][idx,vi].numpy()))))
        checks.append(dict(key=c['key'],region=r,point_count=int(use.sum()),independent_CPU_grid_sample_vs_saved_FP64_aggregation_rounded_FP32_max_abs=errors,scope='One fixed first matched region per cohort; original cached feature numeric read only after global geometry lock; no new model or match. CPUgrid_sample coordinates/interpolationFP32 versus productionFP64.'))
    # Original source resize convention, mask kept independently of filled depth values.
    c=cfg['cases'][0];sp=d.load(c['source_pair']);v=sp['common_valid'][0];z=sp['depth_sensor'][0]
    for name,value in [('depth_sensor',z),('common_valid',v)]:
        pil=np.array(Image.fromarray(value.numpy()).resize((210,210),Image.Resampling.NEAREST));actual=torch.nn.functional.interpolate(value[None,None].float(),size=(210,210),mode='nearest-exact')[0,0].numpy();source_checks.append(dict(field=name,PIL_vs_nearest_exact_max_abs=float(np.max(np.abs(pil.astype(np.float64)-actual)))))
    # Analytic camera round trip for one existing valid anchor scene, no correspondence regeneration.
    c=cases[0];payload=d.load(c['a']['image_path']);K=payload['K_metric'].double();cam=payload['c2w'].double();valid=payload['valid'];ix=torch.nonzero(valid.flatten()).flatten()[:256];uv=torch.stack(((ix%210).double()+.5,(ix//210).double()+.5),-1);depth=payload['target'].flatten()[ix].double();pixels=torch.cat((uv,torch.ones(len(uv),1,dtype=torch.float64)),-1);world=((pixels@torch.linalg.inv(K).T)*depth[:,None])@cam[:3,:3].T+cam[:3,3];local=(world-cam[:3,3])@torch.linalg.inv(cam[:3,:3].T);p=local@K.T;back=p[:,:2]/p[:,2:];error=float((back-uv).abs().max())
    d.write(d.OUT/'interface_validation.json',dict(complete=True,feature_interpolation=checks,source_resize=source_checks,camera_analytic_roundtrip_max_abs210pixels=error,extra_CPU_feature_numeric_file_reads=4,model_forward=0,gradients=0,renders=0,geometry_matching_not_changed=True))

def figures(summary,places,selected,cfg):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,ConnectionPatch
    out=d.OUT/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for cohort,color,y in [('query_source','#367f92',0),('source_source','#b38540',1)]:
        for j,key in enumerate(('K','R')):
            val=summary['cohorts'][cohort]['metrics'][key];axs[0,j].plot(val['interval95'],[y,y],color=color,lw=2);axs[0,j].scatter(val['mean'],y,color=color)
        vv=[r for r in places if r['cohort']==cohort];axs[1,0].scatter([r['K'] if r['K'] is not None else np.nan for r in vv],range(12),color=color,marker='o' if y==0 else 'x',label=cohort)
        counts=[r['view_pairs_defined'] for r in vv];axs[1,1].barh(np.arange(12)+(-.18 if y==0 else .18),counts,height=.32,color=color,label=cohort)
    for j,key in enumerate(('K','R')):
        axs[0,j].set_yticks([0,1],['query-source (primary)','source-source (auxiliary)']);axs[0,j].axvline(0 if key=='K' else 1,color='.5',lw=.8);axs[0,j].set_xlabel(key);axs[0,j].set_title('a  Regional identity K' if key=='K' else 'b  Direct substitution cost R',loc='left')
    ids=[r['id'] for r in places if r['cohort']=='query_source']
    for ax in axs[1]:ax.set_yticks(range(12),ids);ax.invert_yaxis()
    axs[1,0].axvline(0,color='.5',lw=.8);axs[1,0].set_xlabel('Place K (undefined auxiliary omitted)');axs[1,0].set_title('c  Place variation',loc='left');axs[1,0].legend(frameon=False,fontsize=8)
    axs[1,1].set_xlabel('Defined view pairs (primary /16; auxiliary /6)');axs[1,1].set_title('d  Fixed coverage; no replacement',loc='left');axs[1,1].legend(frameon=False,fontsize=8)
    fig.suptitle('Cross-view cached DINO content · geometry-only matched four-vector swaps\nUsed development, place bootstrap; auxiliary intervals are not an interaction')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'outcomes.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(len(selected),2,figsize=(12,3.25*len(selected)),constrained_layout=True)
    for ri,row in enumerate(selected):
        case=next(c for c in cfg['cases'] if c['key']==row['key']);g=d.load(d.read(d.OUT/'geometry'/f'{row["key"]}.json')['data_path']);pt=g['anchor_point_indices'].numpy().astype(int);pa=np.stack(((pt%210)+.5,(pt//210)+.5),-1);pid=(pa[:,1]*32/210).astype(int)*32+(pa[:,0]*32/210).astype(int);pb=g['projected_b'].numpy()
        for vi,im in enumerate((case['a'],case['b'])):
            payload=d.load(im['image_path']);rgb=payload['rgb'] if im['cohort']=='query' else payload['rgb'][im['source_index']];ax=axs[ri,vi];ax.imshow(rgb.permute(1,2,0).numpy());uv=pa if vi==0 else pb
            for reg,col,label in [(row['r'],'#00e8e8','r'),(row['t'],'#ff9933','t')]:
                p=uv[pid==reg]*448/210;ax.scatter(p[:,0],p[:,1],s=8,c=col,alpha=.75,linewidths=0);lo=p.min(0);hi=p.max(0);ax.add_patch(Rectangle(lo,*(hi-lo),edgecolor=col,fill=False,lw=1));ax.text(lo[0],max(0,lo[1]-4),label,color=col,weight='bold',bbox=dict(facecolor='black',alpha=.6,pad=1,edgecolor='none'))
            ax.set_title(f"{case['id']} {'query' if im['cohort']=='query' else 'source'}{im['frame']} — same r/t world points");ax.set_xticks([]);ax.set_yticks([])
        axs[ri,0].set_ylabel(f"{row['label']}\nK={row['K']:.3f}; R={row['R']:.3f}\nr={row['r']}, t={row['t']}\nE_same={row['E_same']:.4f}\nE_swap={row['E_swap']:.4f}; E_within={row['E_within']:.4f}",fontsize=8)
    fig.suptitle('Actual same-world-point correspondences and matched distinct regions\nCyan r / orange t; identical vectors used by same, swapped and within-view distances')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'correspondences.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':score()
