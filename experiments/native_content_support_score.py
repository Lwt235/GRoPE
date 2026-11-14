"""Place-paired fixed-probe statistics and actual sensitivity/area figures."""
import os
os.environ['MPLCONFIGDIR']='/tmp/liwt_native_content_support_mpl'
import time,csv,resource
import numpy as np
import torch
from experiments import native_content_support as d

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    return x
def mean(v):
    v=[x for x in v if x is not None and np.isfinite(x)];return float(np.mean(v)) if v else None
def interval(values):
    values=np.array([v for v in values if v is not None],dtype=np.float64);n=len(values)
    if not n:return dict(mean=None,interval95=None,places=0)
    rng=np.random.default_rng(20261114);boot=values[rng.integers(0,n,(5000,n))].mean(1) if n>=2 else None
    return dict(mean=float(values.mean()),interval95=np.percentile(boot,[2.5,97.5]).tolist() if boot is not None else None,places=n)

def score():
    torch.set_num_threads(1);start=time.monotonic();cfg=d.config();tokens=[];images=[];input_identities=[];max_independent=0.
    for case in cfg['images']:
        folder=d.OUT/'images'/case['key'];done=d.read(folder/'complete.json');assert done['complete'] and done['parameter_versions_unchanged'] and done['parameter_grads_none'];assert d.sha(folder/'energy.npy')==done['energy_sha256'] and d.sha(folder/'layout.pt')==done['layout_sha256']
        e=np.load(folder/'energy.npy',mmap_mode='r');layout=d.load(folder/'layout.pt');assert e.shape==(16,4,1024) and np.isfinite(e).all() and (e>=0).all()
        commits=[d.json.loads(s) for s in (folder/'probe_commits.jsonl').read_text().splitlines()];assert len(commits)==64 and len({(r['token_index'],r['probe']) for r in commits})==64
        for r in commits:assert d.hashlib.sha256(np.asarray(e[r['token_index'],r['probe']]).tobytes()).hexdigest()==r['energy_sha256']
        for ti,u in enumerate(d.TOKENS):
            energy=np.mean(e[ti],axis=0);total=float(energy.sum());self_fraction=float(energy[u]/total) if total>0 else None;defined=bool(layout['read_defined'][ti]);masks={a:layout['regions'][a][ti].numpy().copy() for a in ('G','S','P')};Cs={};probeCs={}
            for a in masks:
                if defined:assert masks[a][u] and int(masks[a].sum())==layout['areas'][ti] and abs(float(layout['densities'][a][ti].sum())-1)<2e-6
                else:assert not masks[a].any()
                masks[a][u]=False;Cs[a]=float(energy[masks[a]].sum()/total) if defined and total>0 else None;probeCs[a]=[float(v[masks[a]].sum()/v.sum()) if defined and v.sum()>0 else None for v in e[ti]]
            if Cs['G'] is not None:
                assert -1e-12<=Cs['G']+self_fraction<=1+1e-12
                # Independent Torch scalar reduction on already saved energies.
                et=torch.from_numpy(np.array(e[ti])).mean(0);c=float(et[torch.from_numpy(masks['G'])].sum()/et.sum());max_independent=max(max_independent,abs(c-Cs['G']))
            tokens.append(dict(key=case['key'],id=case['id'],place=case['place'],cohort=case['cohort'],frame=case['frame'],token_index=ti,token=u,row=u//32,column=u%32,read_defined=defined,total_gradient_energy=total,total_derivative_zero=total==0,self_fraction=self_fraction,area=layout['areas'][ti] if defined else None,area_fraction=layout['areas'][ti]/1024 if defined else None,C=Cs,CG_CP_pp=100*(Cs['G']-Cs['P']) if Cs['G'] is not None else None,CG_CS_pp=100*(Cs['G']-Cs['S']) if Cs['G'] is not None else None,CS_CP_pp=100*(Cs['S']-Cs['P']) if Cs['S'] is not None else None,outside_G_fraction=1-self_fraction-Cs['G'] if Cs['G'] is not None else None,probe_C=probeCs,probe_total_gradient_energy=e[ti].sum(-1).tolist(),probe_CG_CP_pp=[100*(a-b) if a is not None else None for a,b in zip(probeCs['G'],probeCs['P'])]))
        rr=tokens[-16:];images.append(dict(**case,defined_read_tokens=sum(r['read_defined'] for r in rr),zero_derivative_tokens=sum(r['total_derivative_zero'] for r in rr),measured_tokens=16,metrics={k:mean([r[k] for r in rr]) for k in ('CG_CP_pp','CG_CS_pp','CS_CP_pp','self_fraction','outside_G_fraction','area_fraction')},probe_CG_CP_pp=[mean([r['probe_CG_CP_pp'][p] for r in rr]) for p in range(4)]))
        payload=d.load(case['image_path']);rgb=payload['rgb'] if case['cohort']=='query' else payload['rgb'][case['source_index']];input_identities.append(dict(key=case['key'],image_path=case['image_path'],actual_preprocessed_RGB_sha256=d.hashlib.sha256(rgb.numpy().tobytes()).hexdigest(),field_published_sha256=case['published_field_sha256'],old_weight_reference_sha256=cfg['published_model_sha256']));del payload,rgb,layout,e
    places=[];summary={}
    for cohort in ('query','source'):
        for rid in sorted({r['id'] for r in images}):
            rr=[r for r in images if r['id']==rid and r['cohort']==cohort];assert len(rr)==4
            places.append(dict(id=rid,place=rr[0]['place'],cohort=cohort,images=4,images_with_defined_layout=sum(r['metrics']['CG_CP_pp'] is not None for r in rr),defined_tokens=sum(r['defined_read_tokens'] for r in rr),zero_derivative_tokens=sum(r['zero_derivative_tokens'] for r in rr),metrics={k:mean([r['metrics'][k] for r in rr]) for k in ('CG_CP_pp','CG_CS_pp','CS_CP_pp','self_fraction','outside_G_fraction','area_fraction')},probe_CG_CP_pp=[mean([r['probe_CG_CP_pp'][p] for r in rr]) for p in range(4)]))
        pp=[r for r in places if r['cohort']==cohort];tt=[r for r in tokens if r['cohort']==cohort];areas=[r['area'] for r in tt if r['area'] is not None]
        summary[cohort]=dict(metrics={k:interval([r['metrics'][k] for r in pp]) for k in ('CG_CP_pp','CG_CS_pp','CS_CP_pp','self_fraction','outside_G_fraction','area_fraction')},coverage=dict(images=48,tokens=768,defined_read_tokens=sum(r['read_defined'] for r in tt),undefined_read_tokens=sum(not r['read_defined'] for r in tt),zero_derivative_tokens=sum(r['total_derivative_zero'] for r in tt),all_selected_16_reads_undefined_images=[r['key'] for r in images if r['cohort']==cohort and r['defined_read_tokens']==0],places_with_estimable_layout=sum(r['metrics']['CG_CP_pp'] is not None for r in pp)),area_distribution=dict(minimum=min(areas) if areas else None,quantiles=np.percentile(areas,[0,25,50,75,95,100]).tolist() if areas else None,maximum=max(areas) if areas else None,single_patch_regions=areas.count(1),full_image_regions=areas.count(1024)),four_fixed_probe_place_means=[mean([r['probe_CG_CP_pp'][p] for r in pp]) for p in range(4)])
    q=summary['query']['metrics']['CG_CP_pp'];flag=q['mean'] is not None and q['mean']>=1 and q['interval95'] is not None and q['interval95'][0]>0
    d.write(d.OUT/'token_measurements.json',clean(tokens));d.write(d.OUT/'image_measurements.json',clean(images));d.write(d.OUT/'place_measurements.json',clean(places));d.write(d.OUT/'summary.json',clean(dict(complete=True,cohorts=summary,bootstrap=dict(replicates=5000,seed=20261114,unit='Paired places only; token energy first averages4probes, then valid tokens equally within image, defined images equally within place, places equally. Conditional on four fixed probes; no token/probe pseudoreplication.',undefined='Undefined read or zero total derivative ->layout metricNA, neverzero. Their input sensitivity/self statistics retained; no replacement.'),independent_saved_energy_scalar_max_abs=max_independent)))
    d.write(d.OUT/'method_decision.json',dict(query_G_P_at_least1pp_and_positive95lower=flag,query_G_S=summary['query']['metrics']['CG_CS_pp'],source_auxiliary_G_P=summary['source']['metrics']['CG_CP_pp'],PE_or_Native_utility_established=False,independent_confirmation=False,scientific_branch_reserved_for_original_Astra=True,automatic_old_reader_training=False))
    d.write(d.OUT/'input_identities.json',dict(images=input_identities,model=dict(path=cfg['model_path'],published_sha256=cfg['published_model_sha256'],new_large_weight_hash=False),source_fields='Original12J fields shared by all layout controls; no field modified.',source_code={str(p):d.sha(p) for p in (d.ROOT/'experiments').glob('native_content_support*.py')}))
    with (d.OUT/'token_measurements.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['key','place','cohort','frame','token','read_defined','total_derivative_zero','total_gradient_energy','self_fraction','area','CG','CP','CS','CG_CP_pp','CG_CS_pp','CS_CP_pp','outside_G_fraction','probe_CG_CP_pp'])
        for r in tokens:w.writerow([r[k] for k in ('key','place','cohort','frame','token','read_defined','total_derivative_zero','total_gradient_energy','self_fraction','area')]+[r['C'][a] for a in ('G','P','S')]+[r[k] for k in ('CG_CP_pp','CG_CS_pp','CS_CP_pp','outside_G_fraction','probe_CG_CP_pp')])
    defined=[r for r in tokens if r['cohort']=='query' and r['CG_CP_pp'] is not None];undefined=[r for r in tokens if r['cohort']=='query' and not r['read_defined']];selected=[dict(min(defined,key=lambda r:r['CG_CP_pp']),label='Largest negative G-P'),dict(max(defined,key=lambda r:r['CG_CP_pp']),label='Largest positive G-P'),dict(tokens[0],label='Fixed first query/token')]
    if undefined:
        original=original_coverage(cfg);zero_keys={x['key'] for x in original['images'] if x['J_whole_grid_zero_threshold_reference']};candidates=[r for r in undefined if r['key'] in zero_keys and r['key'] not in {x['key'] for x in selected}] or [r for r in undefined if r['key'] not in {x['key'] for x in selected}] or undefined;selected.append(dict(candidates[0],label='Original zero-grid reference / undefined read'))
    d.write(d.OUT/'figure_selection.json',clean(dict(selected=selected,negative_query_tokens=sorted(defined,key=lambda r:r['CG_CP_pp']),rule='Complete primary-query extrema by fixed mainquantity, first fixed token and first undefined read. No model/data/threshold selection.')))
    figures(summary,places,selected)
    d.write(d.OUT/'CPU_scoring_runtime.json',dict(complete=True,body_wall_seconds=time.monotonic()-start,CPU_process_seconds=time.process_time(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_or_gradient_calls=0))
    print(clean(dict(complete=True,query=summary['query'],source=summary['source'])),flush=True)

def figures(summary,places,selected):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=d.OUT/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for cohort,color,offset in [('query','#447c8c',-.12),('source','#b08a57',.12)]:
        for i,k in enumerate(('CG_CP_pp','CG_CS_pp','CS_CP_pp')):
            v=summary[cohort]['metrics'][k];axs[0,0].plot(v['interval95'],[i+offset]*2,color=color);axs[0,0].scatter(v['mean'],i+offset,color=color,label=cohort if i==0 else None)
        vv=[r for r in places if r['cohort']==cohort];axs[0,1].plot([r['metrics']['CG_CP_pp'] for r in vv],range(len(vv)),marker='o',ms=3,color=color,label=cohort)
        areas=[r['area_fraction']*1024 for r in readtokens() if r['cohort']==cohort and r['area_fraction'] is not None];axs[1,0].hist(areas,bins=np.linspace(0,1024,21),histtype='step',color=color,label=cohort)
        for i,k in enumerate(('self_fraction','outside_G_fraction')):
            v=summary[cohort]['metrics'][k];axs[1,1].plot(np.array(v['interval95'])*100,[i+offset]*2,color=color);axs[1,1].scatter(v['mean']*100,i+offset,color=color)
    axs[0,0].set_yticks([0,1,2],['G-P','G-S','S-P']);axs[0,0].axvline(0,color='.6',lw=.8);axs[0,0].axvline(1,color='.7',ls=':',lw=.8);axs[0,0].set_xlabel('Extra total input sensitivity captured (pp)');axs[0,0].set_title('a  Same read and area; self excluded',loc='left');axs[0,0].legend(frameon=False)
    axs[0,1].set_yticks(range(12),[r['id'] for r in places if r['cohort']=='query']);axs[0,1].invert_yaxis();axs[0,1].axvline(0,color='.6',lw=.8);axs[0,1].set_xlabel('G-P (pp)');axs[0,1].set_title('b  Individual places',loc='left')
    axs[1,0].set_xlabel('Region patch count (identical G/P/S per token)');axs[1,0].set_ylabel('Defined tokens');axs[1,0].set_title('c  Area distribution; no threshold repair',loc='left');axs[1,0].legend(frameon=False)
    axs[1,1].set_yticks([0,1],['Own patch (all measured tokens)','Outside G (defined reads only)']);axs[1,1].set_xlabel('Total input sensitivity (%)');axs[1,1].set_title('d  Locality limits (place-paired 95%)',loc='left');fig.suptitle('Frozen single-image DINO input support · fixed four probes · all used development')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'outcomes.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(len(selected),5,figsize=(15,3*len(selected)),constrained_layout=True)
    for i,r in enumerate(selected):
        case=next(c for c in d.config()['images'] if c['key']==r['key']);payload=d.load(case['image_path']);rgb=payload['rgb'] if case['cohort']=='query' else payload['rgb'][case['source_index']];energy=np.load(d.OUT/'images'/r['key']/'energy.npy',mmap_mode='r')[r['token_index']].mean(0);e=energy/energy.sum() if energy.sum()>0 else energy;layout=d.load(d.OUT/'images'/r['key']/'layout.pt');heat=np.log10(np.maximum(e,1e-12)).reshape(32,32)
        axs[i,0].imshow(rgb.permute(1,2,0).numpy());axs[i,0].add_patch(Rectangle((r['column']*14,r['row']*14),14,14,edgecolor='#ffff00',fill=False,lw=1.5));axs[i,0].set_ylabel(f"{r['label']}\n{r['id']},q{r['frame']},u{r['token']}\nG-P={format(r['CG_CP_pp'],'.4f') if r['CG_CP_pp'] is not None else 'NA'} pp",fontsize=8)
        for j in range(1,5):
            im=axs[i,j].imshow(heat,vmin=-6,vmax=0,cmap='magma');axs[i,j].add_patch(Rectangle((r['column']-.5,r['row']-.5),1,1,edgecolor='#ffff00',fill=False,lw=1))
            if j>=2:
                arm=('G','P','S')[j-2];mask=layout['regions'][arm][r['token_index']].numpy().reshape(32,32)
                if mask.any():axs[i,j].contour(mask,levels=[.5],colors=['#55e6e6'],linewidths=.65);axs[i,j].set_xlabel(f"{arm}: K={r['area']}, C={r['C'][arm]:.4f}")
                else:axs[i,j].set_xlabel(arm+': undefined read; no replacement')
            else:axs[i,j].set_xlabel(f"Self={r['self_fraction']:.3f}; log10 fraction" if r['self_fraction'] is not None else 'Total derivative zero')
        for j,ax in enumerate(axs[i]):
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(['Original full RGB / token','Actual sensitivity','G region','P region','S region'][j])
    fig.colorbar(im,ax=axs[:,1:].ravel().tolist(),shrink=.6,label='log10 input-patch derivative-energy fraction (clipped [-6,0])');fig.suptitle('Actual fixed-probe input sensitivity; cyan region boundary, yellow self patch\nAll G/P/S regions have equal area; C excludes the common self patch')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'cases.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def readtokens():return d.read(d.OUT/'token_measurements.json')
def original_coverage(cfg):
    records=[]
    for case in cfg['images']:
        if case['cohort']=='query':
            prior=next(x for x in d.read(d.J/'preparation'/f'{case["id"]}.json')['coverage'] if x['query']==case['frame']);zero=prior['new_zero_all'];oldzero=prior['old_zero_all'];authority=str(d.J/'preparation'/f'{case["id"]}.json')
        else:
            prior=d.read(d.J/'formation'/f'{case["id"]}.json')['final'][case['source_index']];zero=prior['zero_fraction']==1;oldzero=None;authority=str(d.J/'formation'/f'{case["id"]}.json')
        records.append(dict(key=case['key'],cohort=case['cohort'],J_whole_grid_zero_threshold_reference=zero,old_pre_J_query_zero_reference=oldzero,authority=authority,threshold='Original effectivealpha<=1e-8 over all1024patches; distinct from current exact-zero read of16 selected tokens.'))
    result=dict(images=records,all_images_retained=True,no_new_render_or_score=True);d.write(d.OUT/'original_zero_coverage.json',result);return result
if __name__=='__main__':score()
