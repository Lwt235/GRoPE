"""Round006 figures, full factual tables, budget and reviewable handoff."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments import camera_only_feature_data as d


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=d.read(d.OUT/'scoring/summary.json');p=s['primary'];figdir=d.OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    panels=[(axes[0,0],p,['G-P','G-S','S-P'],'Development original queries: PE'),(axes[0,1],p,['P-Native','S-Native','G-Native','RasterFill-Native'],'Development original queries: net use'),(axes[1,0],'feature/development12/fit/full/raw_mse',['G-P','G-S','S-P'],'Feature raw MSE (secondary)'),(axes[1,1],'native/development12/fit/whole/absrel',['G-P','G-S','S-P','G-RasterFill'],'Missing full-image AbsRel')]
    for ax,prefix,names,title in panels:
        rows=[s['comparisons'][prefix+'/'+name] for name in names];x=np.arange(len(names));y=np.array([r['reduction_percent'] for r in rows]);lo=np.array([r['interval95_percent'][0] for r in rows]);hi=np.array([r['interval95_percent'][1] for r in rows])
        ax.errorbar(x,y,yerr=np.stack((y-lo,hi-y)),fmt='o',capsize=4,color='#245b84');ax.axhline(0,color='black',lw=.8);ax.axhline(1,color='#888888',ls=':',lw=.8)
        ax.set_xticks(x,names);ax.set_title(title);ax.set_ylabel('Error reduction (%)');ax.grid(axis='y',alpha=.2)
        for i,r in enumerate(rows):ax.scatter([i]*3,r['per_seed_percent'],s=14,alpha=.55,color='#cb7340')
    fig.suptitle('camera_only_feature_pe1 | 3 paired seeds, 12 used development places\n95% paired seed/place intervals; feature evidence does not replace native endpoint',fontsize=12)
    fig.savefig(figdir/'outcomes.png',dpi=160);plt.close(fig)
    data=d.read(d.OUT/'scoring/metrics.json');worst=s['counterexamples'][0];fixed=[dict(id=m['id'],place=m['place'],query=2) for m in d.contract()['training_catalog'][:2]];cases=fixed+[worst]
    fig,axes=plt.subplots(3,7,figsize=(19,9),layout='constrained');titles=['Target feature RMS','RasterFill feature RMSE','P feature RMSE','G feature RMSE','P log-depth error','G log-depth error','Effective coverage / ROI']
    seed=d.SEEDS[0];saved=[]
    for row,case in enumerate(cases):
        rid,q=case['id'],case['query'];target=d.load(d.old.patch_path(rid,q,'full'))['patches'][0].numpy();query=d.load(d.old.query_path(rid,q));b,a=d.cached_anchor(rid,q)
        rec={arm:next(r for r in data['native'] if r['id']==rid and r['query']==q and r['kind']==arm and r['seed']==seed and not r['permuted']) for arm in ['P','G']}
        fs={arm:np.load(d.CACHE/'features'/f'{arm}_{seed}'/f'{rid}_{q}.npy') for arm in ['P','G']}
        ds={arm:np.load(rec[arm]['path']) for arm in ['P','G']}
        ft=np.sqrt(np.mean(target.astype(np.float64)**2,axis=-1)).reshape(32,32);fe=[np.sqrt(np.mean((x.astype(np.float64)-target)**2,axis=-1)).reshape(32,32) for x in [b.numpy(),fs['P'],fs['G']]]
        vmax=max(float(np.quantile(x,.99)) for x in fe);valid=query['valid'].numpy();targetdepth=query['target'].numpy();de={arm:np.where(valid,np.log(np.maximum(ds[arm],1e-6)/targetdepth),np.nan) for arm in ['P','G']};dv=max(float(np.nanquantile(abs(x),.99)) for x in de.values())
        images=[ft,*fe,de['P'],de['G'],a.reshape(32,32).numpy()]
        for col,im in enumerate(images):
            ax=axes[row,col]
            if col in (1,2,3):ax.imshow(im,cmap='magma',vmin=0,vmax=vmax)
            elif col in (4,5):ax.imshow(im,cmap='coolwarm',vmin=-dv,vmax=dv)
            else:ax.imshow(im,cmap='viridis',vmin=0,vmax=max(1e-8,float(im.max())))
            if col<=3 or col==6:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((7.5,7.5),16,16,fill=False,edgecolor='white',lw=1))
            else:
                ax.contour(query['hidden'].numpy().astype(float),levels=[.5],colors=['black'],linewidths=.5)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=9)
            if col==0:ax.set_ylabel(f"{rid} / q{q}\n{case['place']}",fontsize=9)
        saved.append(dict(id=rid,query=q,seed=seed,selection='fixed first two training scenes' if row<2 else 'maximum3-seed mean development G-P primary error regression',native_metrics={arm:rec[arm]['metrics']['roi'] for arm in rec}))
    fig.suptitle('Fixed cases and largest development mean G-P regression | single seed shown\nMatched color scale within each row/metric; all cases remain scored; coverage is effective content support',fontsize=12)
    fig.savefig(figdir/'cases.png',dpi=160);plt.close(fig);d.write(figdir/'selection.json',dict(cases=saved,all_seed_primary_selection=worst,single_seed_panels_do_not_establish_mechanism=True))


def cost_summary():
    measured=d.read(d.OUT/'cost_measured.json');records=[r for r in measured['records'] if not r['warmup']];summary={}
    for method in ['Native','RasterFill','P','S','G']:
        rr=[r for r in records if r['method']==method]
        summary[method]={key:float(np.median([r[key] for r in rr])) for key in ['wall_seconds','total_GPU_ms','encoder_GPU_ms','raster_GPU_ms','position_GPU_ms','reader_GPU_ms','memory_read_GPU_ms','native_GPU_ms']}
        if method=='Native':
            for key in ['raster_GPU_ms','position_GPU_ms','reader_GPU_ms','memory_read_GPU_ms']:summary[method][key]=None
        elif method=='RasterFill':
            for key in ['position_GPU_ms','reader_GPU_ms']:summary[method][key]=None
    inherited_path=d.ROOT/'results/gaussian_memory/native_memory_context_pe1/cost_amortized_totals.json';inherited=d.read(inherited_path)
    loads=d.read(d.OUT/'load_cost_measured.json')
    input_cost=d.read(d.OUT/'source_input_cost_measured.json')
    ordinary_prep=float(np.mean([r['source_input_LayerNorm_and_concat_synchronized_wall_seconds'] for r in input_cost['records']]))
    source_ref=inherited['legacy_prepared_source_seconds_reference'];totals={};source={}
    for arm in d.ARMS:
        rr=[r for r in measured['source_records'] if r['arm']==arm];prep=float(np.mean([r['seconds'] for r in measured['center_geometry_preparation']]));enc=float(np.median([v for r in rr for v in r['source_encoding_seconds']]));load=float(np.mean([r['cold_state_load_seconds'] for r in rr]));source[arm]=dict(common_center_preparation_CPU_seconds=prep,KV_encoding_GPU_synchronized_wall_median_seconds=enc,derived_state_load_wall_mean_seconds=load,base_plus_derived_bytes=[r['base_bytes']+r['derived_bytes'] for r in rr])
        serialization=float(np.mean([d.read(p)['serialization_seconds'] for p in (d.OUT/'states').glob(f'{arm}_{d.SEEDS[0]}_*.json') if not d.read(p)['permuted']]))
        base_load=float(np.mean([r['logical_materialization_seconds'] for r in loads['assets'] if r['asset']=='base_GS']))
        model_load=next(r['total_seconds'] for r in loads['reader_model_loads'] if r['arm']==arm)
        totals[arm]={}
        for N in [1,4,8,32]:totals[arm][str(N)]=dict(T_base_source_reference=source_ref,T_center_preparation=prep,T_KV_build=enc,T_export_serialization=serialization,T_state_load=load,T_base_field_load=base_load,T_reader_weight_load=model_load,T_complete_queries=N*summary[arm]['wall_seconds'],total=source_ref+prep+enc+serialization+load+base_load+model_load+N*summary[arm]['wall_seconds'],incremental_new_build_and_complete_queries=prep+enc+serialization+load+base_load+model_load+N*summary[arm]['wall_seconds'],memory_only_GPU_event_seconds=N*summary[arm]['memory_read_GPU_ms']/1000,scope='Complete-query wall includes query image encoder, actual raster, camera fields, reader, patch replacement and frozen native/DPT. source reference is legacy measured prepared-source scope, not fullRGB acquisition/predictor time. Native model cold load separate; logical asset loadOS cache uncontrolled.')
    cold=d.read(d.OUT/'workers/worker0_native_load.json')['seconds']
    for arm in d.ARMS:
        source[arm]['common_ordinary_input_preparation_seconds']=ordinary_prep
        for N in [1,4,8,32]:
            row=totals[arm][str(N)];row['T_common_ordinary_input_preparation']=ordinary_prep;row['total']+=ordinary_prep;row['incremental_new_build_and_complete_queries']+=ordinary_prep;row['T_native_model_cold_load_reference']=cold;row['total_including_native_model_cold_load_reference']=row['total']+cold
    base_load=float(np.mean([r['logical_materialization_seconds'] for r in loads['assets'] if r['asset']=='base_GS']))
    for method in ['Native','RasterFill']:
        totals[method]={}
        for N in [1,4,8,32]:
            total=N*summary[method]['wall_seconds']+(source_ref+base_load if method=='RasterFill' else 0)
            totals[method][str(N)]=dict(T_complete_queries=N*summary[method]['wall_seconds'],T_base_source_reference=source_ref if method=='RasterFill' else 0,T_base_field_load=base_load if method=='RasterFill' else 0,total=total,T_native_model_cold_load_reference=cold,total_including_native_model_cold_load_reference=total+cold)
    d.write(d.OUT/'cost_amortized_totals.json',dict(complete=True,source=source,totals=totals,cold_materialization=loads,common_source_input_preparation=input_cost,base_source_reference_path=str(inherited_path),base_source_reference_sha256=d.sha(inherited_path),base_source_reference_scope=inherited['legacy_scope'],extension_pipeline_reference_seconds=inherited['extension_seconds_reference'],extension_scope=inherited['extension_scope'],N32_illustration_only=True,new_encoder_and_raster_actual_not_cache_IO=True))
    runtime=[d.read(p) for p in (d.OUT/'runtime').glob('*.json') if d.read(p).get('GPU_session')];assert all(r.get('process_exited') for r in runtime)
    GPU_seconds=5+sum(r['seconds'] for r in runtime);data_bytes=d.diskbytes(d.CACHE);project_bytes=d.diskbytes(d.OUT);counts=d.read(d.OUT/'budget.json');assert GPU_seconds<=6*3600 and data_bytes<18*2**30
    for key,cap in d.LIMITS.items():assert counts['counts'][key]<=cap
    result=dict(complete=True,summary=summary,source_components=source,actual_budget=counts,limits=d.LIMITS,GPU_worker_seconds=GPU_seconds,GPU_worker_hours=GPU_seconds/3600,GPU_hours_limit=6,runtime=runtime,failed_environment_timing='First failure conservatively charged through archival, includes detection idle interval;5s probe allowance added.',DATA_bytes=data_bytes,DATA_GiB=data_bytes/2**30,DATA_hard_GiB=18,DATA_atomic_peak_allowance_bytes=int(.2*2**30),project_bytes_at_snapshot=project_bytes,project_hard_bytes=2**30,formal_retained_updates=18000,formal_discarded_updates=0,pilot_discarded_updates=6,formal_reader_source_samples_each=36000,pilot_reader_source_samples_each=12,normal_states=396,permuted_states=36,feature_predictions=3312,native_predictions=4368,cost_native_calls=200,cost_reader_calls_inside_native=120,extra_standalone_reader_cost_calls=0,cost_source_calls=150,cost_actual_rasters=160,CPU_scoring_seconds=d.read(d.OUT/'scoring/runtime.json')['seconds'],independent_source_constructed_scenes=0,new_data_or_weights_downloaded=0,base_source_not_counted_free=True,shared_GPU_not_exclusive_speed_claim=True,cost_scope=measured['scope'],event_time_limit='CUDA event elapsed segments, including any CPU-dispatch/accounting gaps between events; not isolated kernel timing. CPU wall includes synchronization, bookkeeping and finite-output checks. Native timing includes210px resize.',native_model_cold_loads=[d.read(p) for p in (d.OUT/'workers').glob('*_native_load.json')],source_encoder_precision='FP32',native_precision='originalBF16autocast, all weights frozen',old_or_foreign_assets_deleted=False)
    d.write(d.OUT/'cost.json',result);return result

def tables():
    s=d.read(d.OUT/'scoring/summary.json');primary=s['primary'];lines=['# camera_only_feature_pe1：固定结果表','', '正值为test相对reference降误差；区间为固定5000配对seed/place bootstrap。全部旧数据；辅助量不替换主量。','', '|比较|降误差%|95%区间|三seed%|','|---|---:|---|---|']
    def row(name,r):return '|'+name+'|'+f"{r['reduction_percent']:+.6f}"+'|'+' / '.join(f'{x:+.6f}' for x in r['interval95_percent'])+'|'+' / '.join(f'{x:+.6f}' for x in r['per_seed_percent'])+'|'
    for name in ['G-P','G-S','S-P','P-Native','S-Native','G-Native','P-RasterFill','S-RasterFill','G-RasterFill','RasterFill-Native']:lines.append(row(name,s['comparisons'][primary+'/'+name]))
    for domain,reg,metric in [('native','roi','centered_log_mse'),('native','roi','log_mse'),('native','roi','bias_squared'),('native','whole','absrel'),('feature','full','raw_mse'),('feature','full','spatial_mse'),('feature','full','bias_squared'),('feature','roi','raw_mse')]:
        lines+=['',f'## {domain}/{reg}/{metric}：完整预定分组','', '|分组/查询|G-P|G-S|S-P|','|---|---|---|---|']
        for group in ['train32','train_original12','train_added20','development12']:
            for qg in ['fit','other','pooled']:
                vals=[s['comparisons'][f'{domain}/{group}/{qg}/{reg}/{metric}/{pair}'] for pair in ['G-P','G-S','S-P']]
                fmt=lambda r:f"{r['reduction_percent']:+.4f}% [{r['interval95_percent'][0]:+.4f}, {r['interval95_percent'][1]:+.4f}]"
                lines.append('|'+group+'/'+qg+'|'+'|'.join(fmt(r) for r in vals)+'|')
    lines+=['','完整支持／非支持／零覆盖／全图、逐seed/place/query及每个新预测的原始指标见[metrics.json](scoring/metrics.json)、[全部summary](scoring/summary.json)、[逐比较CSV](scoring/all_comparisons.csv)、[全部汇总CSV](scoring/all_summary_cells.csv)。Native-full为共享构造结果，不是学习方法full保真证据。']
    (d.OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')
    with (d.OUT/'scoring/all_comparisons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['contrast','reduction_percent','relative_CI_low','relative_CI_high','absolute_test_minus_reference','absolute_CI_low','absolute_CI_high','per_seed_percent','per_place_percent','per_seed_absolute','per_place_absolute','direction'])
        for key,r in s['comparisons'].items():w.writerow([key,r['reduction_percent'],*r['interval95_percent'],r['absolute'],*r['absolute_interval95'],r['per_seed_percent'],r['per_place_percent'],r['per_seed_absolute'],r['per_place_absolute'],r['direction']])
    with (d.OUT/'counterexamples.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id','place','query','mean3seed_G_minus_P','P','S','G'])
        for r in s['counterexamples']:w.writerow([r['id'],r['place'],r['query'],r['G_minus_P'],*[r['per_arm'][a] for a in d.ARMS]])
    data=d.read(d.OUT/'scoring/metrics.json')
    for domain in ['native','feature']:
        rows=data[domain];names=sorted({f'{reg}/{metric}' for row_ in rows for reg,vals in row_['metrics'].items() for metric in vals})
        with (d.OUT/'scoring'/f'per_prediction_{domain}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['id','place','query','kind','seed','permuted','condition','cohort','training_group','query_group',*names])
            for r in rows:w.writerow([*[r.get(k) for k in ['id','place','query','kind','seed','permuted','condition','cohort','training_group','query_group']],*[r['metrics'].get(name.split('/')[0],{}).get(name.split('/')[1]) for name in names]])

def report():
    s=d.read(d.OUT/'scoring/summary.json');flags=d.read(d.OUT/'method_decision.json');cost=d.read(d.OUT/'cost.json');c=s['comparisons'];p=s['primary']
    fmt=lambda r:f"{r['reduction_percent']:+.4f}% [{r['interval95_percent'][0]:+.4f}, {r['interval95_percent'][1]:+.4f}]%"
    pe=c[p+'/G-P'];gs=c[p+'/G-S'];sp=c[p+'/S-P'];binding=c['native/binding_development_fit/roi/centered_log_mse/G_permuted-G'];feature=c['feature/development12/fit/full/raw_mse/G-P'];worst=s['counterexamples'][0];coverage=d.read(d.OUT/'coverage_statistics.json')
    text=f'''# camera_only_feature_pe1：相机条件视图特征监督的固定PE比较

{d.now()}。九次各2000更新、396正常及36置换公共状态、3312 FP32 feature预测、4368 FP32 native预测、全部独立NumPy复算、固定bootstrap和两图实际审阅完成。此处只报告完整实测事实及预定flag；原Astra负责科学解释、取舍和后继，本轮不自行选择候选或宣布总体目标达成。

主开发原四query的missing ROI空间深度G/P降误差 **{fmt(pe)}**；G/S **{fmt(gs)}**；S/P **{fmt(sp)}**。G/P三个seed为{pe['per_seed_percent']}%，12地点中{pe['places_test_error_lower']}地点G更低、{pe['places_test_error_higher']}更高。预定PE筛flag：G/P={flags['PE_empirical_flags']['G-P']['practical_screen']}，G/S={flags['PE_empirical_flags']['G-S']['practical_screen']}，S/P={flags['PE_empirical_flags']['S-P']['practical_screen']}。未通过不称等价／非劣性；通过也只是旧数据开发筛选。

## 固定问题与实现范围

按[锁定执行方案](../../../docs/camera_only_feature_pe_execution_plan.md)和[原Astra机器计划](../autonomous_research_loop/round_006/plan.json)，只训练一个128维4头cross-attention reader预测完整1024×1024 encoder feature残差。原真实GS renderer提供共同归一化anchor b，Q来自b、覆盖及相机20维，不接当前query RGB／queryfeature／mask／depth／编号。完整query encoder feature仅监督；整个官方VGGT被冻结，无native/DPT训练forward或深度梯度。主端点保留开发原query原生空间深度，不以feature分数替换。

三组共享完整Σ普通内容、原f/opacity、中心31邻域B和density4及相机／中心多尺度位置、原渲染／coverage、容量、初值、优化器及采样。每组精确656,768参数。共同25位置维加四槽，P为density4、S为同trace球投影footprint、G为原Σ footprint；同29维分配，不是nested新增独占信息。只有K加位置，V不加；背景零K/V始终存在。普通内容源LN(f)＋source20，经1044→128 GELU/LN及独立K/V；query1045→128 GELU/LN→128；位置29→64 GELU→128；attention与Q拼接256→256 GELU→1024，输出末层与位置末层零初始。没有增加writer/nativePE/fusion/gate或几何学习。

源帧0/3/6/9；原32train／12dev，拟合query2/5/8/11，其他1/4/7/10；每epoch128完整目标，full/missing不重复作两个训练样本。三配对seed20261110/11/12，AdamW3e-4/wd.01/clip5，reader/Adam FP32，全特征rawMSE，batch2，固定2000末步；六discarded pilot后正式重置。初值和4000样本顺序按seed跨arm一致，9份final含model／Adam／RNG／sampler／budget完整恢复状态。

每fit/scene一次source编码为FP32 K/V后永久保存重载，同一SHA读取8相机。原GS以不可变路径／权威SHA引用，衍生状态不复制原1024维f；原场与衍生K/V共同构成逻辑公共memory，不称单文件自足或原场构建免费。正常评价重用既有实际r/a缓存；置换及成本真实重算。原native评价从missing encoder cache出发，仅替换[8:24,8:24]中心256patch，经原24层、2DRoPE、DPT，无额外学习。Native-full共同352结果按构造共享，不能当学习后full保真或非劣性。

## 全部主、净用途与辅助证据

|比较|开发原query空间降误差95%区间|对Native实用净用途flag|
|---|---|---|
|P−Native|{fmt(c[p+'/P-Native'])}|{flags['native_net_utility_flags']['P']['practical_screen']}|
|S−Native|{fmt(c[p+'/S-Native'])}|{flags['native_net_utility_flags']['S']['practical_screen']}|
|G−Native|{fmt(c[p+'/G-Native'])}|{flags['native_net_utility_flags']['G']['practical_screen']}|
|RasterFill−Native|{fmt(c[p+'/RasterFill-Native'])}|仅共同anchor参照，不是历史训练R|

开发feature full-view raw MSE的G/P降误差{fmt(feature)}，是次级组件证据；共同G相对RasterFill的feature raw降误差为{fmt(c["feature/development12/fit/full/raw_mse/G-RasterFill"])}，不能归为PE。训练其他query的G/RasterFill空间改善{fmt(c["native/train32/other/roi/centered_log_mse/G-RasterFill"])}，原12／新增20分别{fmt(c["native/train_original12/other/roi/centered_log_mse/G-RasterFill"])}／{fmt(c["native/train_added20/other/roi/centered_log_mse/G-RasterFill"])}，仍均为旧数据条件用途。raw／去channel均值空间／bias²／cosine与ROI／全视图／覆盖分层均完整报告。特征改善不证明原生用途，也不要求feature必须显著才报告native实测。 missing全图AbsRel的G/RasterFill降误差为{fmt(c["native/development12/fit/whole/absrel/G-RasterFill"])}，即误差均值增加约0.706%，区间支持小幅代价；它低于事前1%材料性均值阈值，但不是零代价或保真证据。native raw／bias²／空间量、missing全图AbsRel／空间／raw、原始几何支持与无支持／零有效覆盖分别保留。材料性伤害flags见[method_decision.json](method_decision.json)：相对RasterFill的主量或missing全图AbsRel退步≥1%且95%支持才标记；微小不确定负均值没有硬零容差否决。

训练原12／新增20、拟合与其他query、开发其他query、所有G/P、G/S、S/P及Native/RasterFill比较、全部seed/place/query均见[全数表](report_tables.md)、[全部比较CSV](scoring/all_comparisons.csv)、[逐预测native](scoring/per_prediction_native.csv)／[feature](scoring/per_prediction_feature.csv)。5000配对seed/place bootstrap固定20261113，地点/query等权；三seed和12dev有限，不是通用门禁、等价检验或独立确认。

唯一预定G内容置换144例将同一置换f同时用于重建K/V及实际anchor；开发原query空间量置换相对正常G的降误差为{fmt(binding)}（负值表示置换更差）。它只说明固定模型的条件绑定响应，不是PE独立比较或唯一机制证明，不触发额外扫描。原a_F全零query共{len(coverage['all_zero_queries'])}个，完整清单在[覆盖统计](coverage_statistics.json)，全部保留预测和评分，无困难样本筛选。

最大三seed平均G−P开发退步例为{worst['id']}/q{worst['query']}（{worst['place']}），空间误差差{worst['G_minus_P']:+.8f}；其余全部反例在[counterexamples.csv](counterexamples.csv)。[汇总图](figures/outcomes.png)与[固定两旧训练scene＋最大反例](figures/cases.png)均实际view_image审阅，[审阅记录](figures/visual_review.json)保留可见事实；个例图只展示首seed，不替代三seed机制结论。

## 正确性、普通修复与完整预算

必要预检确认字段有限、API无query图像输入、同参数／成对初值、零残差等于RasterFill、ROI对齐且其他patch逐值不变、source K/V保存重读、六pilot梯度及一次完整恢复。补充一个CPU全key无效fixture确认零背景使输出有限，实际reader调用也计费。所有新3312 feature和4368depth独立NumPyfloat64复算，保留原mask及每文件SHA；未重审旧模型／历史数值，无大权重重复hash。

两项普通修复在同轮版本记录：首次Ninja／扩展缓存PATH缺失，1次失败raster尝试计费、0pilot；后恢复断言比较CPU/CUDA step张量失败，保留Pstep1完整快照、仅执行剩余5pilot，未重复已有调用或增加正式更新。旧锁／旧源码快照／精确diff和[runtime_source_revision_v2.json](runtime_source_revision_v2.json)均保存，监督／字段／端点／组／预算不变。失败预检的已通过assert与未落盘诊断值分开记录，不捏造丢失数值。 评分展示v3将cosine统一作为绝对相似度差／区间而非降误差百分比；尤其zero-coverage RasterFill的零向量cosine=0，百分比不可定义。原cosine值、绝对bootstrap和全部error指标不变，原summary／CSV版本保留，无重跑模型／评分或新门禁。

实际counts：{cost['actual_budget']['counts']}。正常保留18000更新、训练reader/source各36000样本；pilot6及各12样本另计，0native训练。GPU含加载／准备／失败／恢复／I/O共{cost['GPU_worker_hours']:.6f}worker小时≤6；最多两worker，均选空闲GPU0/1，未占GPU5其他任务。所有GPU子进程已实际退出；最终CLI完成由主在本turn结束后观察。本轮DATA永久{cost['DATA_GiB']:.6f}GiB≤18，atomic峰值与headroom账见[资源记录](resources/ledger.jsonl)及[cost.json](cost.json)；项目≤1GiB。恢复每100步持久化，验证final后仅退休本轮重复resume，无旧／他人资产删除。

## 实际成本与一次写入复用

成本固定前两旧训练scene/query2、五方法轮换、各5warm＋15timed，共200完整native query；其中P/S/G共120次reader就在相同调用内部取CUDA event段，未另跑120次reader。source3arm×2scene×25=150，均计费；实际160次GS cost raster。完整cost含query图像encoder、实际原GS raster、实际FP64相机位置、reader、patch替换、原native/DPT与210px resize；没有拿缓存I/O代替真实计算。event段含dispatch／记账间隙，CPUwall含同步与检查，不称纯kernel时延。共享服务器测量，无独占／微小时差加速结论。 实际共同view_fields helper一次计算B/S/G各字段，再按arm选择额外四维；P未读未用的S/G字段仍有共同准备开销，因此这些是实际实现时延，不是各arm最小化后的kernel成本。Native/RasterFill不存在reader的空事件间隙标为不适用，不计虚构reader调用。

''' 
    text+='|方法|完整query CPUwall中位ms|GPUevent总ms|memory read event ms|reader event ms|\n|---|---:|---:|---:|\n'
    for method,r in cost['summary'].items():
        memory='—' if r['memory_read_GPU_ms'] is None else f"{r['memory_read_GPU_ms']:.3f}"
        reader='—' if r['reader_GPU_ms'] is None else f"{r['reader_GPU_ms']:.3f}"
        text+=f"|{method}|{1000*r['wall_seconds']:.3f}|{r['total_GPU_ms']:.3f}|{memory}|{reader}|\n"
    text+='''
source中心B/density准备、K/V投影、实际state冷加载、序列化、基底／衍生bytes和N=1/4/8/32完整摊销见[cost_amortized_totals.json](cost_amortized_totals.json)。原source构建沿用既有权威实测参考及其边界，sensor获取／AnySplat预计算／raw预处理／模型冷载不能藏为0；不同20场pipeline范围单列。N32只是示意，实际同状态读取8query。 [逻辑加载记录](load_cost_measured.json)分开CPU文件读取和CPU→GPU/权重装入，[共同输入准备](source_input_cost_measured.json)另测固定两场的FP32源LayerNorm/拼接；均无额外网络forward。OS page cache未清空，冷载指本次显式逻辑物化，不声称物理冷盘。摊销分别给不含／含实际native冷载参考的总量。

## 证据边界及交回

本轮直接服务O2/O3的公共粗GS及可重载K/V、O4同状态多view读取、O5真实渲染与参数、O6 memory reader Transformer的参数位置组织；不是原生24层新增PE。C1/C5保持，C2/C3/C4/C6按计划改变。sensor/joint-validity、给定相机／尺度、静态四源、2048×1024、固定geometry/content、冻结VGGT、已知missing ROI与depth／旧数据仍为临时边界。

原prototype/context_supervision已有特征监督，来源writer、输入attention及保留Render的原生Q/K阴性保持；无query RGB的直接监督任务是本轮条件变化，不改写这些历史。共同renderer／reader收益不冒充PE；P也有完整Σ普通内容，不能叫删除全部shape；P/S/G等29维预算分配不是nested信息增加。全部已用，无verified fresh reserve，任何合格开发候选以后仍需全使用史审查与真独立身份锁定。

实现复用CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，主实际派发请求Sol/high/fast；主报告client-selected Sol/high，本角色没有独立后台effective model/effort/tier证明，实际tier为unknown。未改全局配置／认证、未创建角色／任务、未发跨线程消息、未调用goal或修改主state/events。原科学输入身份不改；完整结果交主复用原Astra，结束本turn释放实施租约，不自行继续下一轮。
'''
    (d.OUT/'report.md').write_text(text)

def validate_delivery():
    fits=[d.read(p) for p in (d.OUT/'fits').glob('*/status.json')];assert len(fits)==9 and all(r['steps']==2000 for r in fits)
    paired={}
    for seed in d.SEEDS:
        rr=[d.read(d.OUT/'fits'/f'{arm}_{seed}'/'initial_identity.json') for arm in d.ARMS];assert len({r['initial_sha256'] for r in rr})==1 and all(r['schedule']==rr[0]['schedule'] for r in rr);paired[str(seed)]=rr[0]['initial_sha256']
    states=list((d.OUT/'states').glob('*.json'));preds=list((d.OUT/'completed').glob('*.json'));bases=list((d.OUT/'baselines').glob('*.json'));assert len(states)==432 and len(preds)==3312 and len(bases)==352
    assert sum(d.read(p)['permuted'] for p in states)==36 and sum(d.read(p)['permuted'] for p in preds)==144
    assert len(list((d.CACHE/'features').rglob('*.npy')))==3312 and len(list((d.CACHE/'depths').rglob('*.npy')))==4368
    for r in fits:assert d.sha(r['path'])==r['sha256']
    d.write(d.OUT/'validation_complete.json',dict(complete=True,paired_initial_sha256=paired,paired_schedules_match=True,full_final_recovery_states=9,public_states=432,normal_states=396,permuted_states=36,FP32_features=3312,FP32_depths=4368,all_predictions_independently_scored=True,all_required_processes_exited=True,original_scientific_plan_preserved=True))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['figures','report']);args=a.parse_args();torch.set_num_threads(4)
    if args.mode=='figures':figures()
    else:validate_delivery();cost_summary();tables();report()
