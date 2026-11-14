"""Finalize completed round025 artifacts and living links; no GPU, model or new statistics."""
import json,pathlib,hashlib,datetime,time,resource,collections,math
ROOT=pathlib.Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_crossview_content1';ROUND=ROOT/'results/gaussian_memory/autonomous_research_loop/round_025'
start=time.monotonic()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def fmt(v):return f"{v['mean']:.6f} [{v['interval95'][0]:.6f}, {v['interval95'][1]:.6f}]" if v['mean'] is not None else 'NA'
cfg=read(OUT/'configuration.json');assert sha(ROUND/'science_plan.md')==cfg['science_sha256']
summary=read(OUT/'summary.json');pool=read(OUT/'pool_complete.json');assert pool['complete'] and pool['all_children_completion_verified']
terminal=read(OUT/'terminal_process_check.json');assert terminal['all_recorded_GPU_worker_PIDs_absent']
old=OUT/'engineering_revisions/attempt001_nearest_exact';attempts=[('engineering_invalid_nearest_exact',old),('formal_original_PIL',OUT)]
workers=[read(p) for _,base in attempts for p in (base/'workers').glob('*.json')];events=[json.loads(l) for _,base in attempts for p in (base/'calls').glob('*.jsonl') for l in p.read_text().splitlines()]
assert len(workers)==48 and sum(e['kind']=='geometry_pair_complete' for e in events)==528 and sum(e['kind']=='vector_pair_complete' for e in events)==528
gpu=sum(w['elapsed_seconds'] for w in workers);assert gpu==pool['resources']['GPU_worker_seconds'] and gpu<3600
assert all(w.get('actual_wait_returned') or w.get('completion_observed_without_wait') for w in workers)
counts=dict(collections.Counter(e['kind'] for e in events));timings={}
for k in counts:
    v=[e['seconds'] for e in events if e['kind']==k and 'seconds' in e]
    if v:timings[k]=dict(count=len(v),sum_seconds=sum(v),mean_seconds=sum(v)/len(v),minimum_seconds=min(v),maximum_seconds=max(v))
cost=dict(all_attempt_operation_counts=counts,all_attempt_timers=timings,GPU_worker_seconds=gpu,GPU_worker_hours=gpu/3600,
    vector_arithmetic=dict(region_quadruples_total=sum(e.get('region_pairs',0) for e in events),squared_1024D_distances_total=sum(e.get('distances',0) for e in events),world_point_view_interpolations_total=sum(e.get('world_point_view_interpolations',0) for e in events)),
    scope='Worker launched-until-waited elapsed includes CUDA/import/load, CPU matching, vector arithmetic, serialization/hash/I-O and normal polling tails. Two orphan workers additionally charged through confirmed-process-absence (conservative55.05s each). No hidden invalid/recovery work.',
    timer_boundaries={'geometry_pair_complete':'Projection/visibility, region moments, support footprints, matching, CPU greedy, new file write/fsync/hash. Source input load/CUDA/import outside this sub-timer but within worker wall.',
    'feature_materialization':'Existing60feature files per attempt, CPU load and8views/place transferred; no encoder.',
    'vector_pair_complete':'New geometry reload, point aggregation, exact same4vector six distances, write/fsync/hash. Includes I/O; not a pure GPU kernel timer.'},
    CPU_interface_extra_feature_file_reads=8,CPU_interface_scope='Four original feature numeric reads per attempt for fixed first query-source and source-source aggregation checks, after respective global geometry locks. No model or extra feature prediction.',
    zero_operations=dict(training=0,model_forward=0,new_prediction=0,new_gradient=0,render=0,download=0,old_scores_recomputed=0,old_large_weight_hash=0),
    inherited_cost='Original DINO encoding/sensor acquisition/camera derivation occurred in old caches. These are dependencies, not free end-to-end frontend costs; this round measures cached arithmetic only. No native-query/cold-model latency claim.')
cost['attempts']=[]
for label,base in attempts:
    calls=[json.loads(l) for p in (base/'calls').glob('*.jsonl') for l in p.read_text().splitlines()]
    ww=[read(p) for p in (base/'workers').glob('*.json')];cost['attempts'].append(dict(label=label,workers=len(ww),GPU_worker_seconds=sum(w['elapsed_seconds'] for w in ww),counts=dict(collections.Counter(e['kind'] for e in calls)),region_quadruples=sum(e.get('region_pairs',0) for e in calls),source='Full invalid attempt is retained and charged, not a scientific alternative.' if base==old else 'Only this PIL-correct attempt supplies formal scientific statistics.'))
cost['formal_counts']=cost['attempts'][1]['counts'];write(OUT/'cost_actual.json',cost)

places=read(OUT/'place_measurements.json');pairs=read(OUT/'viewpair_measurements.json');queries=read(OUT/'query_measurements.json');sources=read(OUT/'source_measurements.json');quads=read(OUT/'region_pair_measurements.json');gg=read(OUT/'geometry_quality.json');sel=read(OUT/'figure_selection.json')['selected']
negative=[q for q in quads if q['K'] is not None and q['K']<0]
write(OUT/'counterexamples.json',dict(all_negative_region_pairs=sorted(negative,key=lambda x:x['K']),largest_primary_substitution_cost=max((q for q in quads if q['cohort']=='query_source'),key=lambda q:q['R']),minimum_primary_viewpair_K=min((p for p in pairs if p['case']['cohort']=='query_source' and p['K'] is not None),key=lambda p:p['K']),selection=sel,no_exclusion_or_new_subgroup_endpoint=True))
quality={}
for cohort in ['query_source','source_source']:
    rows=[g for g in gg if g['cohort']==cohort];first=rows[0]['stage_counts'];quality[cohort]=dict(stage_counts=[dict(stage=s['stage'],remaining=sum(g['stage_counts'][i]['remaining'] for g in rows),newly_rejected=sum(g['stage_counts'][i]['newly_rejected'] for g in rows)) for i,s in enumerate(first)],
        raw_quadruple_quality_quantiles={k:__import__('numpy').percentile([q[k] for q in quads if q['cohort']==cohort],[0,25,50,75,95,100]).tolist() for k in ['coverage_r','coverage_t','depth_a_r','depth_b_r','view_angle_r_deg','separation_a_px','separation_b_px','separation_world_m','forward_depth_max_m','reverse_depth_max_m','reprojection_a_max_px','reprojection_b_max_px']})
write(OUT/'geometry_quality_summary.json',quality)
oldzero=read(ROOT/'results/gaussian_memory/native_content_support1/original_zero_coverage.json')['images'];zero_records=[]
for image in oldzero:
    if image['cohort']!='query' or not image['J_whole_grid_zero_threshold_reference']:continue
    q=next(x for x in cfg['images'] if x['key']==image['key']);record=next(x for x in queries if x['id']==q['id'] and x['query']==q['frame']);zero_records.append(dict(image_key=image['key'],inherited_reference=image,new_current_query_result=record))
write(OUT/'historical_zero_query_retention.json',dict(reference_scope='Old J GS-read zero flags only, not this round sensor eligibility. Joined without old metric/raster recomputation; all four fixed query×source pairs retained.',records=zero_records))

review=dict(actual_viewed=True,PNG_sha256={n:sha(OUT/'figures'/n) for n in ['outcomes.png','correspondences.png']},PDF_SVG_retained=True,
    observations=['K primary and auxiliary95%intervals are positive. PrimaryR interval crosses1 while auxiliaryR is above1; different place coverage and pair populations preclude an interaction claim.',
    'Coverage panel visibly retains fidelity03 only1/16 mainpair and no auxiliary for00/02/03. All12 primaryplace K are positive, but384 individual primary regional K are negative.',
    'First displayed main correspondence fidelity00 q11/s9 shows two cabinet-edge/wall regions at notably different image positions; its K=.585 andR=1.821 are conditional on given sensor/poses.',
    'Minimum regionalK fidelity05 q5/s6 r235/t302 lies on a tiledwall; K=-.67517,R=3.13609 despite full common-point fractions. Same/swap reversal is retained without claiming descriptor universal failure.',
    'MaximumR fidelity11 q8/s3 r244/t340 lies on a whitewall/door edge; K=.31240,R=17.02271 with small within-region spatialcontrast denominator. Exact raw E displayed.',
    'First auxiliary fidelity01 s0/s3 has wall/decoration/painting-edge appearances whose correspondence is visually ambiguous. Projected depth/pose agreement is not independent semantic groundtruth. It is retained and flagged, not interpreted as proof of physical-surface mismatch or feature instability.',
    'RGB, pairedsame world-sample projections, cyan r/orange t and extrema labels are readable; no effect-selected example removed.'],
    limitation='Worldpoint correspondence is defined by supplied sensor/pose and strict bidirectional tests, not newly verified true surface identity. Low-texture/edges and potential RGB-depth/pose errors limit literal physical-substitution claims.')
write(OUT/'figure_review.json',review)

known_CPU=[read(base/'CPU_scoring_runtime.json') for _,base in attempts];host_records=[read(p) for _,base in attempts for p in (base/'worker_resources').glob('*.json')];largest=sorted((x['peak_host_RSS_bytes'] for x in host_records),reverse=True)[:2]
first_measure=read(next((OUT/'measurements').glob('*.json')))
newdata=[pathlib.Path(read(OUT/'engineering_revisions/source_resize_failure001.json')['old_DATA_directory']),pathlib.Path(first_measure['data_path']).parent.parent]
data_bytes=sum(p.stat().st_size for base in newdata for p in base.rglob('*') if p.is_file());project_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in (ROOT/'experiments').glob('native_crossview_content*.py'))
cpu_start=datetime.datetime.fromisoformat('2026-11-13T23:50:00+00:00');cpu_upper=3*((datetime.datetime.now(datetime.timezone.utc)-cpu_start).total_seconds()+120)
budget=dict(limits=cfg['limits'],GPU_worker_seconds=gpu,GPU_worker_hours=gpu/3600,all_attempts_and_recovery_included=True,
    CPU_measured_scoring_body_seconds=sum(x['body_wall_seconds'] for x in known_CPU),CPU_process_scoring_seconds=sum(x['CPU_process_seconds'] for x in known_CPU),
    CPU_conservative_worker_hour_upper=cpu_upper/3600,CPU_upper_scope='Three simultaneous slots charged continuously from23:50UTC (before implementation dispatch) through delivery+120s. Maximum actual simultaneous workers twoGPU+manager; source-preflight and CPU scoring stages nonoverlap. This overbounds waits/engineering reads/failedmanager/recovery/figure/packaging; not pureCPUutilization.',
    host_RAM=dict(engineering_admission_limit_GiB=32,no_fixed_two_card_limit=True,per_worker_reserved_GiB=3,manager_reserved_GiB=2,actual_max_concurrent_workers=2,
    persisted_max_sampled_concurrent_worker_RSS_bytes=max(pool['resources']['max_observed_concurrent_child_RSS_bytes'],read(old/'pool_complete.json')['resources']['max_observed_concurrent_child_RSS_bytes']),
    sum_two_largest_individual_worker_peak_RSS_upper_bytes=sum(largest),individual_peak_RSS_recorded_for_all48=True,CPU_scoring_peak_RSS_bytes=max(x['peak_host_RSS_bytes'] for x in known_CPU),manager_reserve_GiB=2,
    scope='Per-worker ru_maxrss persists actual lifetime maxima including two lost-parent workers. Sum of largest two bounds workerRSS concurrency; add2GiB manager/packaging reserve. No RAM staging. GPU tensor allocations are separate in worker_resources.'),
    storage=dict(all_attempt_DATA_permanent_bytes=data_bytes,DATA_peak_conservative_bytes=data_bytes+32*2**20,project_permanent_before_packaging_bytes=project_bytes,project_peak_conservative_bytes=project_bytes+64*2**20,
    scope='Includes full invalidv1 and correctedv2, all checkpoints/locks/feature-vector files, scripts and oldattempt score/figures.32MiBDATA reserve for largest atomic payload;64MiBproject reserve for largest JSON atomic/temp, final manifest/report/handoff and three full living docs. No original largeweight/field/data copies.',new_RAM_staging_bytes=0),
    compliant=True,unknown_status_boundary='Two orphan workers completed all22geometry commits and wrote resource completion, then were confirmed absent. Actual exitcodes unavailable aftermanager40361exit1; charged conservative observation tails, never labelledwait0.')
assert data_bytes+32*2**20<4*2**30 and project_bytes+64*2**20<2**30
write(OUT/'execution_budget.json',budget)
write(OUT/'execution_completion.json',dict(all_GPU_and_CPU_science_processes_ended=True,terminal_check='terminal_process_check.json',manager_attempts=[dict(exec_session=44995,actual_exit_code=0,status='v1 engineering-invalid measurements archived'),dict(exec_session=40361,actual_exit_code=1,status='ordinaryRSSrace;88validgeometrypairs retained'),dict(exec_session=94824,actual_exit_code=0,status='resumed only unfinishedpairs')],CPU_scoring=[dict(exec_session=44826,actual_exit_code=0,scope='invalidv1'),dict(exec_session=81670,actual_exit_code=0,scope='formalPIL')],all24formal_workers_completed=True,formal_workers_parent_waited_exit0=22,formal_workers_completed_and_observed_absent_exitcode_unknown=[90298,90299],budget_failure_records=['engineering_revisions/source_resize_failure001.json','engineering_revisions/manager_proc_race002.json'],no_duplicates_of_valid_formal_pair_commits=True))

q=summary['cohorts']['query_source'];s=summary['cohorts']['source_source'];metrics=['| Quantity | primary query–source,12 places | auxiliary source–source,9 places |','|---|---:|---:|']
for k in ['K','R','E_same','E_swap','E_within']:metrics.append(f"| {k} | {fmt(q['metrics'][k])} | {fmt(s['metrics'][k])} |")
table=['| Place | primary defined pairs /16 | defined queries /4 | region pairs | K | R | auxiliary pairs /6 | auxiliary K | auxiliary R |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in [r for r in places if r['cohort']=='query_source']:
    a=next(r for r in places if r['cohort']=='source_source' and r['id']==p['id']);ak=f"{a['K']:.4f}" if a['K'] is not None else 'NA';ar=f"{a['R']:.4f}" if a['R'] is not None else 'NA'
    table.append(f"| {p['id']} {p['place']} | {p['view_pairs_defined']} | {p['queries_defined']} | {p['region_pairs']} | {p['K']:.4f} | {p['R']:.4f} | {a['view_pairs_defined']} | {ak} | {ar} |")
report=f'''# native_crossview_content1 — actual cached cross-view content

2026-11-14. Finite implementation complete, scientific disposition returns to original Astra. **Primary K={fmt(q['metrics']['K'])}**, positive95%lower, supports region-difference signal within the measured sensor/pose-defined scope. **R={fmt(q['metrics']['R'])} crosses1**, so direct substitution remains unresolved. This is the locked regional-signal/substitution-unresolved branch, not proof of stable constant world vectors, PE usefulness, native net utility, GS compression, fresh confirmation or overall success. Geometric coverage and ambiguous projected edge/wall cases materially limit literal same-surface interpretation.

[Unique scientific contract](../autonomous_research_loop/round_025/science_plan.md), SHA256 `{cfg['science_sha256']}`; [analysis handoff](../autonomous_research_loop/round_025/analysis_handoff.json), [available input facts](../autonomous_research_loop/round_025/crossview_input_facts.json), [configuration](configuration.json), [input identities](input_identities.json). No second scientific plan. Reused original gpt-6.1-sol/high request; Fast field unavailable, unrequested/unverified and backend effective configuration unknown.

## Question and fixed objects

The hypothesis is shared regional content across views; competitors are whole-image/place similarity, shared regional differences with view-dependent offsets, and visibility/member/geometry-resampling errors. Round024 ended the local-input-support rationale after G/P failed its locked condition while retaining G/S positive and broad sensitivity. That derivative measurement does not answer cross-view content correspondence. Earlier source-only and native-task field formation supplied reconstruction/training examples without this same-four-vector matched exchange. This round changes the observable and correspondence, without reopening a reader or training family. [Round024 judgment](../autonomous_research_loop/round_024/science_plan.md), [source formation](../native_feature_field_pe1/report.md), [native-task formation](../native_task_writer_pe1/report.md).

Exactly12 original used development places, four sources0/3/6/9 and original full queries2/5/8/11:96 existing images.192query×source and72unordered earlier-anchor source×source viewpairs, all retained. Complete original1024D single-image DINO caches, no re-encoding, PCA, whitening, centering, norm matching or learning. J fields and theirΣ/read rows never select these regions; no field is rebuilt or modified. Query features/sensor are offline measurement only. Existing sensor/common-valid/published camera/RGB and feature identities are referenced from their old small manifests, with no old largeweight or feature-cache rehash. Source-pair and original query payload identities are explicit; all new geometry/vector payloads have checksums.

O2/O3 public-content premise and O4 cross-view reading motivate this C2 measurement. C5 supplied sensor/common_valid/cameras, static four-source scenes and used data are temporary limits. C1/C3/C4/C6 model/field/PE/training stay unchanged; this is not O5GS-compression fidelity or O6increment. All previous positive/negative outcomes and family closures, especially round024, are preserved.

## Geometry locked before feature values

Query target/valid/K_metric are the original210² cache. Source448² depth_sensor and **common_valid** use exactly the original PIL nearest210resize; intrinsics scale210/448. Filled depth1 is never used to reconstruct validity. CachedRGB448 remains unchanged. Pixel centers are edge coordinates(x+.5,y+.5), scaled along original crop to448 and32² featuregrid. Each anchor14²pixel patch footprint defines a region via210grid center membership; source-source anchors use earlier sourceframe.

Each valid anchor point is backprojected to a world point and projected into the target. Nearest target210sensor center must be in-view/valid and agree in depth within max(.02m,.02×measuredDepth). Its measured center/depth backprojects to anchor; the nearest reverse anchor must likewise be valid with the same reverse-depth rule. Both210pixel-center cycle errors must be≤1pixel, with positive/in-frame target/cycle projections. No depth interpolation through invalid values. Feature sampling uses the **original anchor world points**, continuously projected in both images, so both means use exactly the same members and uniform anchor-grid weights. At least16accepted points and at least50%of all footprint grid points are required. These given sensor/pose checks are not independent semantic surface groundtruth.

Region geometry: mean depth in eachview; two sorted principal projected RMS scales in eachview; accepted/grid coverage. Eligible pairs have eachview depthmax/min≤1.25, each of four scalemax/min≤1.5 and coverage difference≤.10. Their **positive-weight bilinear participating patch index sets are disjoint in EACH view**. Stable ascending anchorpatchID greedy selection chooses the closest remaining eligible unused region, no replacement, tiesbyID. Distance is equal-weight mean of seven squared tolerance-normalized differences: two fractional depthratios/.25, four fractional scaleratios/.5, one coveragedifference/.10. Zero/zero scales ratio1, positive/zero infinite; no epsilon or postscore tuning.

In each attempt, all264geometry tables and matching IDs were saved and globally locked **before that attempt's feature numeric phase**. The invalidv1 attempt had already materialized these same feature caches before formalPILv2 geometry was recomputed; this exposure is retained, not presented as pristine first-access prospectivity. The correction follows the original fixedPIL contract and geometry-only code, without feature-dependent matching or threshold changes. [Geometry lock](geometry_lock.json) checks all payload identities; feature-read timestamps are later than lock. EachDATAgeometry file retains all accepted anchor indices/projected target locations/residuals, all1024region stats, qualified/matched IDs and packed interpolation supports. Every invalid/unmatched region remains in the coverage/stage logs; no threshold was relaxed or replacement data used.

## Same four vectors and aggregation

Uniformly bilinear-read all1024channels at those exact common worldpoints in both original featuregrids. Coordinate is32*p210/210−.5, border handling is common and align_corners=False equivalent. FP64 point aggregation, rounded once to savedFP32 regional means; FP64 six distances use those same saved values. Four vectors x=f_a(r), y=f_b(r), u=f_a(t), v=f_b(t) are used unchanged by E_same=.5[d(x,y)+d(u,v)], E_swap=.5[d(x,v)+d(u,y)], E_within=.5[d(x,u)+d(y,v)], d=norm²/1024.

K=(E_swap−E_same)/E_within, R=E_same/E_within. The algebraic difference is(x−u)·(y−v)/1024; it removes common view-offset cross terms, so positiveK does not prove direct original-vector substitution. Every viewpair first equally averages regional E values. Main sources equally within each query, then queries equally within place, then place ratiosK/R; final mean is over places, not a pooled ratio or region-count-weighted mean. Missing sources/queries are NA, with defined counts shown, not zero/replaced. Auxiliary fixed six sourcepairs are equally averaged within each defined place then ratios; no source/query interaction is inferred. Place-balanced mean E rows are reported separately and need not have ratios equal to meanplaceK/R.5000 paired-place bootstrap, seed20261114, conditional on this fixed cache/matchpopulation; regions/pairs are not independent places. A singleplace would get no overallinterval.

## Results, complete coverage and variation

{chr(10).join(metrics)}

Primary108/192viewpairs,18407regionpairs,12/12definedplaces. Auxiliary37/72viewpairs,3548regionpairs,9/12definedplaces; fidelity00/02/03 are undefined auxiliary. Primary37964/196608candidate regions qualify,36814are matched,1150qualifiedunmatched; auxiliary7457/73728qualify,7096matched,361qualifiedunmatched. Remaining regions fail common-member count/coverage; reason counts can overlap and sequential perpoint rejection counts are separately retained.84primary and35auxiliaryviewpairs have no matchedregionpair. ZeroE_within regionalpairs0, zero viewpair denominator0; no zero-spatial-contrast filtering occurred. Primary384negative regionalK, auxiliary93negative; all region-level values are retained even though all defined viewpairK and all primary placeK are positive.

{chr(10).join(table)}

fidelity03's main outcome rests on only oneviewpair/onequery, so12place means do not mean full16pair coverage perplace. Camera/depth/visibility selection narrows the scope; this is not unseen-data confirmation. All four previous J-GS whole-zero query references stay in the fixed query×source list; [historical zero query retention](historical_zero_query_retention.json) joins old flags without recalculating old scores/rasters. Old GS zero and this sensor matching eligibility are distinct.

[Every viewpair](viewpair_measurements.json), [48query records](query_measurements.json), [48source diagnostics](source_measurements.json), [24place/cohort records](place_measurements.json), [all21955regionpair records](region_pair_measurements.json), [CSV](region_pair_measurements.csv), [all477negative regional cases and extrema](counterexamples.json). [Fullgeometry quality](geometry_quality.json) retains stage rejection totals, all-common residual quantiles and matched depth/scale/coverage/viewangle summaries; [summary](geometry_quality_summary.json) reports descriptive raw-regionpair quantiles/separations. They are quality/scope records, not searched scientific subgroups.

## Actual visual review and counterexamples

![Balanced results and coverage](figures/outcomes.png)

![Projected correspondences and extrema](figures/correspondences.png)

Both final PNGs were actually viewed; PDF/SVG and currentSHA identities are in [figure review](figure_review.json). Figure1 keepsK/Rseparate, the R=1reference and three undefinedauxiliary places, including primary1/16coverage atfidelity03. Figure2 shows cyanr/oranget copies of the same sensor/pose-derived world samples. The fixedfirst displayed main case is lexical-keyfirst matched region, not a favorable effect-selected case.

The smallest primary regionalK is **−.675170** at fidelity05 q5/s6,r235/t302, R3.136091, E_same .00942982,E_swap .00739967,E_within .00300687. Both regions have full common-point fractions and lie on a tiledwall; swap beats same and remains a genuine counterexample within the supplied geometry. The largestR is **17.022709** at fidelity11 q8/s3,r244/t340; K.312404, E_same .01382122,E_swap .01407487,E_within .000811928, coverage2/3each. Small same-view regional contrast makes replacement expensive even with positiveK; low-texture whitewall/door-edge context is visible. No small-denominator cases were removed or normalized away.

The firstmain cabinet-edge/wall example hasK.585299,R1.821032. The firstauxiliary fidelity01s0/s3 example hasK.772465,R3.241359; wall/decoration/painting-edge appearances are visually ambiguous under the supplied projections. Depth and reprojection threshold compliance alone cannot prove correct semantic surface correspondence or exclude RGB/depth/pose calibration error. These examples are preserved and flag limits, **not evidence that an obviously wrong match establishes representation instability**. The overall positiveK is therefore conditional regional signal in the fixed sensor/pose matched population; precise physical-point invariance remains unverified. We did not alter geometry to rescue the measure.

## Validation, ordinary failures and recovery

Independent CPU NumPy arithmetic on all21955saved same-four-vector quadruples recomputes allE/K/R; maxGPU-versusCPU E/ratio difference is{read(OUT/'independent_verification.json')['max_E_and_ratio_absolute_difference']:.6g}, exchange/dot identity residual{read(OUT/'independent_verification.json')['max_exchange_dot_identity_difference']:.6g}. All matched geometric constraints, uniqueness/no-replacement and disjoint supports were independently checked; maxsavedmatching-distance difference{read(OUT/'independent_verification.json')['max_geometry_distance_difference']:.6g}. This validates this measurement's arithmetic, not another descriptor/geometry realization or old task scores. [Independent verification](independent_verification.json).

Fixed first matched region percohort CPUgrid_sample versus production FP64aggregate/FP32save differs by at most3.40e−7 in anychannel, with exactly same samples; no materialchange to interpretation and no precision tuning. OriginalPIL versus corrected source resize differs0 in depth and validity. Analytic camera round-trip max8.35e−14pixels. [Interface validation](interface_validation.json), [corrected source preflight](source_resize_preflight_v2.json). No model was instantiated in any stage.

**Two ordinary engineering faults are retained.** Initial v1 used Torch nearest-exact instead of originalPIL nearest; fixedsource interface validation found depth difference.255m and common_valid differences on sampling ties. All264v1geometry and264vector measurements were invalidated and archived **before scientific reporting**, with full558543368B DATA, code/config/locks/figures and107.418382GPUworkerseconds retained and charged. This is not an alternate scientific arm, a K/R-driven selection or a new plan. Corrected v2 uses originalPIL and matched an independentPIL index-remap preflight exactly. [Failure001](engineering_revisions/source_resize_failure001.json).

The firstv2manager(session40361) hit a/proc RSS race: a finishing process had noVmRSSline. It exited1 after88validgeometry tables. Workers90298/90299 finished their22commits each and wrote resource-end records, then were confirmed absent; their actual exitcodes could no longer be waited by their original parent and remain **unknown**, never calledwait0. Launch-to-confirmed-absence55.05s each is charged conservatively. The manager now tolerates disappeared/emptyRSS and its abnormal cleanup signals/waits only owned children. Resume94824 reused all88commits and completed only remaininggeometry then264vector pairs; no validv2pair was repeated. [Failure002](engineering_revisions/manager_proc_race002.json). Perpair lock, atomic fsyncwrite/hash then marker and singlemanager flock prevent duplicatecommits. No actualSIGTERM recovery/power-loss test is claimed.

Originalmanager44995 and archivedCPU44826 exited0; failedmanager40361 exited1; finalmanager94824 and formalCPU81670 exited0. Of24formalworkers,22were actually waited with exit0 and2have independently observed completed+absentstatus with unknownexitcodes. All48workerPIDs across attempts were absent at finalhostcheck; all scientific CPUstages ended. [Completion](execution_completion.json), [terminal hostcheck](terminal_process_check.json). GPU4/7 were used concurrently while6/0–3/5 were externallybusy; no artificial defaulttwocardcap and no externalprocess modified. No oldround023/otherqueue was polled or restarted.

## Actual operations, cost and budget

Across both attempts:48CUDA arithmetic-worker starts,528geometric pairmeasurements,528vector pairmeasurements,120GPU-worker cachedfeaturefile reads,24feature-materialization scene loads.43900regionalquadruples,263400squared1024Ddistances,6657022worldpoint/view interpolations. Formal attempt alone:264geometry+264vectorpairs and21955quadruples (18407main/3548auxiliary),60featurefile reads/96viewfeatures materialized. CPU interface checks add4numericfeature reads perattempt,8total; no new encoder or image. Prepared depth/mask/camera loads and graph construction are included in full workerwall. All phases perform0training/forward/newprediction/gradient/render/download/oldscore-recompute/oldweight-hash.

**GPUworkercharge{gpu:.6f}s={gpu/3600:.8f}h≤1h**, including invalidattempt, imports/CUDA/loading/geometry/CPUmatching/serialization/hash/I-O/recovery and conservative lost-parent observationtails. Combined sub-timer sums: geometry{timings['geometry_pair_complete']['sum_seconds']:.6f}s, featurematerialization{timings['feature_materialization']['sum_seconds']:.6f}s, vectorpairconstruction{timings['vector_pair_complete']['sum_seconds']:.6f}s. They include indicated CPU/I-O and are not isolated kernels; workeroverheads and sourceinput loading remain charged outside sub-timers. [Exactcost ledger](cost_actual.json). Cached original DINO/sensor/camera formation is an inherited dependency, not a free acquisition or a new end-to-end nativequery timing. No native/DPT pass or amortized inference speed is claimed.

CPUscoring bodywall sums{budget['CPU_measured_scoring_body_seconds']:.6f}s; both actualCPUprocess usage records total{budget['CPU_process_scoring_seconds']:.6f}s. Additional manager/preparation/recovery/interface/packaging costs are conservatively overbounded by{budget['CPU_conservative_worker_hour_upper']:.6f}workerh (3slots continuously since23:50UTC through delivery+120s), not calledpureCPUutilization. CPU2threads/worker; only2safe GPUs concurrently, with liveRAM/disk guards. Per-worker lifetime peakRSS preserved for all48workers; sum of two largest peaks{sum(largest):,}B bounds concurrentworkerRSS, plus2GiB manager/packaging reserve. Persisted concurrentRSS samplemax{budget['host_RAM']['persisted_max_sampled_concurrent_worker_RSS_bytes']:,}B, CPUscoringpeak{budget['host_RAM']['CPU_scoring_peak_RSS_bytes']:,}B. Actual GPU allocationpeaks are in worker_resources. Engineering admission32GiB host,3GiB/worker+2GiBmanager, no RAM staging; cardcount adapts to safeoccupancy and independent remainingtasks.

AllnewDATA including invalidv1+formalv2 **{data_bytes:,}B**, conservative atomicpeak+32MiB={data_bytes+32*2**20:,}B≤4GiB. Project including fulloldattempt/code/finalstats beforepackaging{project_bytes:,}B; conservative+64MiB={project_bytes+64*2**20:,}B≤1GiB, covering finalreport/manifest/livingdocs and largesttemporary. Both versions remain; no oldmodel/field/data deletion or systemGLIBCchange. [Budget ledger](execution_budget.json) and finalmanifest expose actualidentities/counts/limits; GPUusage did not expand withcardcount.

The finite content-carrier measurement and validation are complete. Regional signal exceeds fixed same-view matched swaps under the given geometric scope; main direct substitution remains uncertain and some correspondence/coverage limits remain. Auxiliary R>1 is preserved without changing the primarybranch or claiming interaction. Implementation correctness, contentmechanism, GS/PE/nativeutility and freshconfirmation remain distinct. Original Astra decides the single scientificdisposition; this implementation neither declares overall success/pauses research nor opens a successor or old family.
'''
(OUT/'report.md').write_text(report)

status='**round025实施完成，待原Astra处置（2026-11-14）：native_crossview_content1。** 主K=.7182 [.6589,.7700]支持所测范围的区域差分；R=.8163 [.6145,1.0273]跨1，直接替换未决。108/192主对、12地点，辅助37/72对、9地点；完整覆盖／负例及投影视觉歧义保留，非PE／Native／fresh证据。'
links='[报告](../results/gaussian_memory/native_crossview_content1/report.md) · [唯一合同](../results/gaussian_memory/autonomous_research_loop/round_025/science_plan.md) · [交接](../results/gaussian_memory/native_crossview_content1/implementation_handoff.json)。O2–O4/C2内容取证、C5匹配范围；round024及全部历史关闭／正负不改，整体未达、未暂停，无外部阻塞。'
docs=[ROOT/'docs'/n for n in ['research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md']]
for path in docs:
    lines=path.read_text().splitlines();assert 'round025' in lines[3];lines[3]=status+links;text='\n'.join(lines)+'\n'
    if path.name=='research_outline_alignment.md':
        oldrow=next(l for l in lines if l.startswith('| [round025跨view内容'))
        newrow='| [round025跨view内容报告](../results/gaussian_memory/native_crossview_content1/report.md)，实施完成待Astra | O2–O4；C2内容性质／C5匹配范围 | 全264对测量；主K .7182 [.6589,.7700]，R .8163 [.6145,1.0273]；108/192主对与全部零覆盖／负例／几何视觉限制保留。 | 区域信号条件有支持，直接替换未决；无训练／模型前向／新预测，非PE／Native／fresh；旧关闭保持。 |'
        text=text.replace(oldrow,newrow).replace('固定一次跨view内容测量，比较共同可见区域的实际值，而非重测单图导数','已完成一次跨view内容测量，主K有条件支持、R跨1，见[完整报告](../results/gaussian_memory/native_crossview_content1/report.md)；没有重测单图导数')
    path.write_text(text)
write(OUT/'CPU_delivery_runtime.json',dict(complete=True,body_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,no_model_GPU_calls=True,scientific_statistics_not_changed=True))
files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','implementation_handoff.json')]+list((ROOT/'experiments').glob('native_crossview_content*.py'))+docs
manifest=dict(project_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))],DATA_files=[],scope='Allnewproject/code/livingdocversions plus bothnewDATAattempts. NewDATA hashes reused from durablepairmarkers and CPUintegrityverification, no repeatedoldweight/featurehash. Manifest/handoff exclude circularselves.')
for label,base in attempts:
    for folder in ('geometry','measurements'):
        for p in (base/folder).glob('*.json'):
            r=read(p);manifest['DATA_files'].append(dict(path=r['data_path'],bytes=r['data_bytes'],sha256=r['sha256'],attempt=label,integrity_source='Durableworkerhash and completedindependentCPU reload/verification'))
write(OUT/'artifact_manifest.json',manifest)
handoff=dict(status='implementation_complete',experiment='native_crossview_content1',contract=dict(path=str(ROUND/'science_plan.md'),sha256=cfg['science_sha256']),report=dict(path=str(OUT/'report.md'),sha256=sha(OUT/'report.md')),manifest=dict(path=str(OUT/'artifact_manifest.json'),sha256=sha(OUT/'artifact_manifest.json')),
    artifacts={n:str(OUT/n) for n in ['summary.json','execution_budget.json','cost_actual.json','execution_completion.json','terminal_process_check.json','independent_verification.json','interface_validation.json','figure_review.json','configuration.json','input_identities.json','counterexamples.json']},
    main_result=dict(K=q['metrics']['K'],R=q['metrics']['R'],locked_branch=summary['criterion']['R_relation_to1'],defined_viewpairs=108,fixed_viewpairs=192,regional_pairs=18407,places=12),auxiliary=s['metrics'],
    budget=dict(GPU_worker_hours=gpu/3600,DATA_permanent_bytes=data_bytes,project_peak_conservative_bytes=project_bytes+64*2**20),
    engineering_failures=['OriginalsourcePILnearest mismatch; v1fullyretained/countedinbudget, nofinalstatisticsuse.','ManagerprocRSSrace;88validgeometrycommitsreused; twochildrenfinished/observedabsentbutactualexitcodesunknown.'],
    unresolved_material_scientific_changes=[],limitations=['Given sensor/pose/visibilityproxy and visuallyambiguous edge/wallprojections limit real physical-surface interpretation.','MainRintervalcrosses1; auxiliary9placeR>1 is separate, notinteraction.','Twoorphanworkerexitcodesunknown; all48PIDsabsent and completedartifacts/resourcesverified, conservativeobservationtailcharged.','Geometrylockprecedesfeaturephase perattempt; priorinvalidv1alreadyaccessed samecaches, disclosed without claimingpristinefirstaccess.','Useddevelopment only, noGScompression/PE/nativeutility/freshclaim.'],all_scientific_CPU_GPU_processes_ended=True,original_model_request='gpt-6.1-sol/high',Fast_requested_or_verified=False,backend_effective_unknown=True,successor_dispatched=False)
write(OUT/'implementation_handoff.json',handoff)
print(json.dumps(dict(report_sha256=handoff['report']['sha256'],manifest_sha256=handoff['manifest']['sha256'],handoff_sha256=sha(OUT/'implementation_handoff.json'),GPU_worker_hours=gpu/3600,DATA_bytes=data_bytes,project_files=len(manifest['project_files']),DATA_files=len(manifest['DATA_files'])),indent=2))
