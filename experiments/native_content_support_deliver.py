"""CPU-only packaging of completed round023 evidence; never reruns measurements."""
import collections, datetime, hashlib, json, os, pathlib, resource, time

ROOT = pathlib.Path('/home/liwt/GRoPE')
OUT = ROOT/'results/gaussian_memory/native_content_support1'
ROUND = ROOT/'results/gaussian_memory/autonomous_research_loop/round_023'
start = time.monotonic()
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p, x): pathlib.Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def fmt(v): return f"{v['mean']:+.4f} [{v['interval95'][0]:+.4f}, {v['interval95'][1]:+.4f}]"

cfg=read(OUT/'configuration.json');summary=read(OUT/'summary.json');decision=read(OUT/'method_decision.json')
assert sha(ROUND/'science_plan.md')==cfg['science_sha256']
pool=read(OUT/'pool_complete.json');assert pool['complete'] and pool['all_children_actual_wait'] and pool['completed_images']==96 and not pool['failures']
workers=[read(p) for p in (OUT/'workers').glob('*.json')]
assert len(workers)==5 and all(w['exit_code']==0 and w['actual_wait_returned'] for w in workers)
assert len(list((OUT/'images').glob('*/complete.json')))==96
events=[json.loads(line) for p in (OUT/'calls').glob('*.jsonl') for line in p.read_text().splitlines()]
counts=collections.Counter(e['kind'] for e in events)
assert counts==dict(CUDA_access=4,encoder_load=4,actual_raster=96,density_layout=96,preflight_cached_reference_forward=1,input_gradient_forward=96,VJP_started=6144,VJP_complete=6144)
timers={}
for k in counts:
    vv=[e['seconds'] for e in events if e['kind']==k and 'seconds' in e]
    if vv: timers[k]=dict(count=len(vv),sum_seconds=sum(vv),mean_seconds=sum(vv)/len(vv),minimum_seconds=min(vv),maximum_seconds=max(vv))
gpu=sum(w['elapsed_seconds'] for w in workers if w['phase']!='prepare')
wall=sum(w['elapsed_seconds'] for w in workers)
cost=dict(actual_calls=dict(counts),timers=timers,GPU_worker_seconds=gpu,
    timer_boundaries={'density_layout':'Includes field load/transfer, actual_raster, projection/density/region construction; raster is nested, do not double count.',
    'VJP_complete':'Backward + FP64 patch energy reduction + CPU transfer + memmap flush/fdatasync + probe journal fsync; joint elapsed wall, not pure CUDA backward kernel. No original I/O-only sub-timer.',
    'encoder_load':'DINO-only instantiate, checkpoint mmap/select/strict load, CUDA move and synchronize; no full VGGT instance.',
    'worker':'Imports/loading/CUDA/checkpoint access/forward/backward/layout/I-O and actual wait observation tails all charged.'},
    separate_IO_calibration='io_cost_calibration.json; fixed first query/source CPU write/fsync/logical reload, not concurrent original write timing, no cold-cache or exclusive-hardware claim.',
    inherited_formation={'report':str(ROOT/'results/gaussian_memory/native_feature_field_pe1/report.md'),'scope':'Existing J source-only 300-step field formation, historical mean9.879s/place after initial targets/field materialization. Inherited dependency, not newly run or free. Source encoder/geometry preparation/cold loading outside that historical fit timer are not zero or newly estimated.'},
    amortization={'new_forward_wall_per_image_seconds':timers['input_gradient_forward']['sum_seconds']/96,
    'joint_VJP_wall_per_image_seconds':timers['VJP_complete']['sum_seconds']/96,
    'new_layout_wall_per_image_seconds':timers['density_layout']['sum_seconds']/96,
    'source_formation_once_per_place':'Inherited J formation reused for all eight images and all layouts; does not become a new measurement or zero-cost acquisition.',
    'scope':'Arithmetic means of actual fixed96 image workloads under three concurrent safe GPUs, not exclusive latency or a new native query benchmark. Each image pays one new forward, one real read and64VJP; no cached energy advertised as a complete query.'})
write(OUT/'cost_actual.json',cost)

review=dict(actual_viewed=True,versions='Final v2 PNGs viewed after CPU refinement actual exit0; corresponding PDF/SVG retained.',
    images={n:sha(OUT/'figures'/n) for n in ['outcomes.png','cases.png']},
    observations=['Main query G-P interval crosses0 and mean is below+1pp; query G-S interval is positive and remains separate.',
    'Place variation and long region-area tail remain visible. Self fraction uses all measured tokens; outsideG uses defined reads only, explicitly labelled.',
    'Largest negative fidelity06 query2 token900: P covers the sensitive left band better; G-P=-37.8345pp at identical119patch area.',
    'Largest positive fidelity06 query11 token404: G captures a diagonal sensitive band; G-P=+63.4209pp at identical70patch area. An extreme is not a place average.',
    'Fixed first token and distinct original zero-grid query show actual nonzero sensitivity and undefined read, with no replacement regions. Cyan boundaries/yellow self and log10 fraction clip[-6,0] are legible.'],
    correction='First figure selection duplicated fixed and undefined examples. CPU-only v2 uses a distinct original zero-grid image and clarifies selected16-undefined versus original full-grid-zero. No metrics/probes/model/region threshold changed.')
write(OUT/'figure_review.json',review)
write(OUT/'process_completion.json',dict(manager=dict(exec_session=69074,pid=60998,actual_exit_code=0,all_children_actual_wait=True),
    workers=workers,CPU_postprocessing=[dict(exec_session=92741,actual_exit_code=0,evidence='CPU_scoring_runtime.json'),dict(exec_session=94627,actual_exit_code=0,evidence='CPU_refinement_runtime.json'),dict(exec_chunk='9687b6',actual_exit_code=0,evidence='independent_metric_verification.json'),dict(exec_chunk='efe1c3',actual_exit_code=0,stage='initial CPU packaging; current final evidence packaging does not recompute statistics')],
    ordinary_failure='engineering_revisions/cpu_figure_lookup_v2_failure.json and cpu_packaging_v2_failure.json; one CPU import exit1 and one CPU packaging idempotence exit1; no GPU/model/VJP/render or discarded metrics.',
    owned_measurement_PIDs_absent_at_UTC='2026-11-13T23:29:18Z',host_ps_and_NVML_checked=True,foreign_jobs_untouched=True,
    GPU_work_actual_end=pool['ended_utc'],no_resumption_or_new_stage=True))

q=summary['cohorts']['query'];s=summary['cohorts']['source'];pp=read(OUT/'place_measurements.json');zz=read(OUT/'original_zero_coverage.json')
rows=['| Place | query G−P pp | query G−S pp | defined query tokens /64 | source G−P pp | defined source tokens /64 |','|---|---:|---:|---:|---:|---:|']
for x in [r for r in pp if r['cohort']=='query']:
    y=next(r for r in pp if r['cohort']=='source' and r['id']==x['id'])
    rows.append(f"| {x['id']} {x['place']} | {x['metrics']['CG_CP_pp']:+.4f} | {x['metrics']['CG_CS_pp']:+.4f} | {x['defined_tokens']} | {y['metrics']['CG_CP_pp']:+.4f} | {y['defined_tokens']} |")
metricrows=['| Quantity | query mean [95%] | source auxiliary mean [95%] |','|---|---:|---:|']
for k in ['CG_CP_pp','CG_CS_pp','CS_CP_pp','self_fraction','outside_G_fraction','area_fraction']:
    metricrows.append(f"| {k} | {fmt(q['metrics'][k])} | {fmt(s['metrics'][k])} |")
zeroq=[r['key'] for r in zz['images'] if r['cohort']=='query' and r['J_whole_grid_zero_threshold_reference']]
zeros=[r['key'] for r in zz['images'] if r['cohort']=='source' and r['J_whole_grid_zero_threshold_reference']]
now=datetime.datetime.now(datetime.timezone.utc)
elapsed=(now-datetime.datetime.fromisoformat('2026-11-13T23:13:00+00:00')).total_seconds()+120
cpu_upper=4*elapsed
assert cpu_upper<4*3600
sources=list((ROOT/'experiments').glob('native_content_support*.py'))
docs=[ROOT/'docs'/n for n in ['research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md']]
size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in sources)
budget=dict(limits=cfg['limits'],GPU_worker_seconds=gpu,GPU_worker_hours=gpu/3600,
    measured_prepare_plus_GPU_child_wall_seconds=wall,
    CPU_worker_hours_conservative_upper=cpu_upper/3600,
    CPU_upper_scope='Four concurrent worker-slots charged continuously from23:13UTC (earlier than dispatch), through packaging time plus120s reserve. Covers manager, GPU-worker CPU, prepare, scoring/refinement, inspections/failed helper and idle gaps; not pure CPU usage. Actual stage times retained. Reasoning is not an extra server worker.',
    host_RAM=dict(conservative_admitted_working_set_GiB=20,bound='At most3 workers×6GiB plus2GiB manager. DINO-only1.218GB CPU parameters plus at most1.218GB touched selected mmap weights, libraries/largest-layer temporaries and small geometry/energy buffers; no full VGGT model or checkpoint copy. Preparation and CPU postprocessing do not overlap GPU workers.',
    concurrent_observed_RSS_snapshot_bytes=4668710912,snapshot_UTC='2026-11-13T23:19:35Z',
    peak_scoring_RSS_bytes=read(OUT/'CPU_scoring_runtime.json')['peak_RSS_bytes'],peak_refinement_RSS_bytes=read(OUT/'CPU_refinement_runtime.json')['peak_RSS_bytes'],
    instrumentation_limit='GPU-phase childRSS sampled3s but only last queue snapshot persisted; no independently saved fine-grained whole-phase peak. The observed4.35GiB is not claimed as peak;20GiB is the conservative admission bound, below32GiB.'),
    storage=dict(permanent_bytes_before_final_packaging=size,conservative_peak_bytes=size+32*2**20,
    reserve_scope='32MiB covers report/manifest/handoff, all three living-doc full sizes, largest atomic own artifact and temporary I/O calibration. No RAM staging/data copies/large weights added. Own output currently monotonic except documented tiny I/O temp removal.',
    new_DATA_bytes=0,new_RAM_staging_bytes=0),all_limits_respected=True,
    failures_counted='CPU helper import exit1 and packaging idempotence StopIteration exit1 counted in conservative CPU upper; zero failed/discarded GPU measurements or fitted updates; original valid image/probe commits never repeated.')
assert budget['storage']['conservative_peak_bytes']<3*2**30
write(OUT/'execution_budget.json',budget)

report=f'''# native_content_support1 — completed fixed input-support measurement

2026-11-14. Implementation complete; scientific disposition returns to original Astra. Query **G−P={fmt(q['metrics']['CG_CP_pp'])} percentage points**. This fails the locked mean≥+1pp and positive95%lower criterion. Query **G−S={fmt(q['metrics']['CG_CS_pp'])}pp** is separately positive conditional evidence against the spherical layout, and is preserved. Source G−P={fmt(s['metrics']['CG_CP_pp'])}pp is auxiliary and uncertain. No PE utility, native net utility, fresh confirmation or overall success is established. No old reader/training was reopened.

[Unique scientific contract](../autonomous_research_loop/round_023/science_plan.md), SHA256 `{cfg['science_sha256']}`; [analysis handoff](../autonomous_research_loop/round_023/analysis_handoff.json), [representation facts](../autonomous_research_loop/round_023/representation_facts.md), [configuration](configuration.json), [method flags](method_decision.json). This implements that contract; it is not a second scientific plan. Original requested implementation identity gpt-6.1-sol/high is reused; interface has no Fast/service-tier field, Fast unrequested/unverified, backend effective configuration unknown.

The hypothesis was that physical fullΣ gives an informative support layout for actual stored DINO content beyond strong centers. Competing explanations include own-patch dependence, broad contextual dependence, sufficient center layout and source-rendering freedom inΣ. Unlike round011/012 reconstruction/PE/native measurements, round008 semantics or round022 sampling labels, this measures a new local input derivative. O2/O3 shared J fields and O4/O5 real read define the objects; O6 receives design evidence only, C2 adds a content-property measurement and C4 the matched layout controls. All old positive/negative outcomes and closures remain. Old data, single-image DINO, supplied camera/sensor, static four sources and learned J formation are scope limits, not universal requirements.

## Actual object, precision and controls

All twelve original used development places, sources0/3/6/9 and full original queries2/5/8/11:96 unique images,48query+48source. The frozen output is the complete1024D single-image `aggregator.patch_embed` DINO `x_norm_patchtokens`, before VGGT's subsequent24 frame/global pairs. We strictly loaded all304,372,736 DINO parameters from the published checkpoint by mmap; no full VGGT instantiated, no weight update. Original RGB0..1 at448², original ImageNet normalization, eval, original BF16 autocast, inputFP32, energy reductionFP64. The original SDPA path was usable unchanged. Model published identity `{cfg['published_model_sha256']}` was inherited without rehashing the large checkpoint. [Input identities](input_identities.json) record actual RGB bytes and twelve old field identities; all layouts share each field read-only.

The fixed16 tokens are Cartesian rows/columns4,12,20,28 (lower/right member of the even center pair, fixed before measurements). Four shared Rademacher vectors seed20261114 have entries±1/32, norm1, [saved probes](probes.pt). Each token/probe VJP differentiates its channel projection w.r.t original unmasked RGB. Square and sum all RGB/pixels in each14² input patch, multiply by1024 for the Frobenius estimator, then average four energies before fractions. The scale cancels fractions. This is a four-probe local first-order approximation, not exact Jacobian, attention weights, finite occlusion or cross-view invariance. Per-probe variation is retained without extra probes or independent-model claims.

For each measured image one actual original GS raster yields W. Multiply the selected rows by has_content and normalize once; this **same row** is used by G/S/P. G uses full ordinary physicalΣ, S traceΣ I/3, P equally mixes1/8/16/31 nearest valid-center second moments about the same slot center with unit²×1e−6I ridge. Centers, mixture spread between centers, actual visibility, camera and weights are common; P is not a total removal of shape from the renderer. Project each by the original perspective Jacobian, near.01/far100 and common+.3patch²I. FP64 Gaussian log-density mixtures are normalized once over the image grid, not per component. G includes self and the smallest density-ranked other-patch set reaching95% total image-grid mass; S/P include self and exactly G's count, stable flattened-index ties. No area/threshold/probe/field tuning; no learning/capacity or extra image inputs for G. Layout/regions/read rows are saved for all96images.

`C_X` is nonself energy inside the X region divided by **all** input energy. Four-probe energies combine first, defined tokens equally within image, defined images equally within place, then all12places equally. Missing layout is NA, never0 or replacement; its input sensitivity/self statistics remain.5000 paired-place bootstrap draws, seed20261114, conditional on the fixed probes; tokens/probes are not independent places. The criterion is explicitly conjunctive; neither component passed for G/P. Original Astra owns disposition, with no automatic follow-on fit.

## Full outcomes and heterogeneity

{chr(10).join(metricrows)}

First three quantities are percentage points; remaining rows are fractions. Self uses all measured nonzero-derivative tokens, whereas outsideG/area use defined reads. **These cohort means have different populations and must not be added to make one exact mass partition.** Per defined token self+C_G+outsideG=1. Query self is39.1455% and outsideG21.2101%; source39.6471% and18.4550%. The substantial outside mass limits a local-support account even where G wins.

{chr(10).join(rows)}

Query region count min/Q25/median/Q75/Q95/max={q['area_distribution']['quantiles']}; source={s['area_distribution']['quantiles']}. No region is only self or all1024patches. Query equal-place/image area mean6.2985%, source5.1569%; raw token area histogram is separately shown. Broad tails and same-area controls prevent greater area from being counted as shape advantage.

Four fixed-probe separate query G−P place means={q['four_fixed_probe_place_means']}; source={s['four_fixed_probe_place_means']}. These are ratios per probe averaged after, whereas the primary takes the ratio after energy averaging; they need not average to the primary. They describe probe variation, not a second main endpoint or four independent replications. Complete1536 token rows,96image rows and24place/cohort rows are [token JSON](token_measurements.json), [CSV](token_measurements.csv), [image measurements](image_measurements.json), [place measurements](place_measurements.json) and [summary](summary.json). Every negative query token is retained and sorted in [figure selection](figure_selection.json).

## Undefined reads, original zeros and actual figures

Query defined616/768, undefined152; source defined687/768, undefined81. Zero-total-derivative tokens0 in both; **all1536 selected tokens still measured**. All12places have an estimable layout. Query images with all selected16reads undefined: {', '.join(q['coverage']['all_selected_16_reads_undefined_images'])}. Source: {', '.join(s['coverage']['all_selected_16_reads_undefined_images'])}. This is not the entire1024-patch grid being empty.

Original J full-grid-zero references, inherited from saved metadata at effectivealpha≤1e−8, query: {', '.join(zeroq)}; source: {', '.join(zeros)}. Current selected-read undefined uses exact effective W sum0, a different predicate. [Original zero coverage](original_zero_coverage.json) preserves all96 image flags/authorities, including old pre-J flags. No new coverage raster or old score was computed for this lookup. Full-grid-zero images remain and have nonzero DINO derivatives; they receive no substitute support layout.

![Outcomes](figures/outcomes.png)

![Actual cases](figures/cases.png)

Both final PNGs were actually viewed, with matching PDF/SVG retained and hashes in [figure review](figure_review.json). Outcomes show the main interval crossing0, the separate positive G/S interval, place variation, long area tail and explicitly different self/outside populations. Cases are the complete-query largest negative and positive, fixed first query/token and a distinct original zero-grid image; log10 energy fraction clip[−6,0], cyan region boundaries and yellow self.

Largest negative `fidelity_06_query_2`, token900: **−37.8345pp**, C_G=.30744,C_P=.68578,C_S=.34412,119patches,self=.10834. P follows the sensitive left band better than G's right/lower extension. Largest positive `fidelity_06_query_11`, token404: **+63.4209pp**, C_G=.63443,C_P=.000219,C_S=.000242,70patches,self=.16513; G captures a diagonal sensitive band. These two extremes in the same place do not substitute for place means. Fixed `fidelity_00_query_2` token132 is undefined with self=.27645; original zero-grid `fidelity_02_query_8` token132 is undefined with self=.43356 and visibly nonzero wider sensitivity. No invented G/P/S contour appears for undefined reads.

## Validation, unique execution and ordinary failure

Preflight was the first formal image, not an extra measurement. One extra no-grad forward matched its old1024×1024 cached features with max/RMS difference0; input-grad and no-grad outputs also matched exactly. [Preflight evidence](preflight_cache.json). Frozen parameter versions and no parameter gradients were checked for every image. Per-image flock plus a unique manager lock prevent duplicate work; every8192-byte probe payload is memmap-flushed/fdatasync'd, hashed and journal-fsynced before commitment. SIGTERM preserves the current committed probe and resumes only uncommitted work. No actual interruption/restart was needed, so this is supported durable-interface behavior, not a claimed destructive power-loss test.

CPU scoring reloaded every energy/layout SHA and6144 unique probe journals, checked finite nonnegative energies/regions/equal area and recomputed G scalars independently with Torch versus NumPy: max absolute difference **{summary['independent_saved_energy_scalar_max_abs']:.6g}**. A separate CPU verifier uses Python math.fsum directly on all four saved probe entries for **all G/P/S**, rebuilds image/place means and5000paired-place intervals: C maximum difference4.44e−16 and endpoint maximum difference1.33e−15pp, [full verification](independent_metric_verification.json). These validate saved-energy metrics, not another encoder Jacobian realization. No authoritative results were changed and no old depth/feature scores were recomputed. Actual manager exec69074/PID60998 exit0 waited all children: prepare60999, preflight61047/GPU4, shards61230/GPU4,61231/GPU6,61232/GPU7 all exit0. Three safe cards were used simultaneously, with live CUDA/NVML/foreign-PID/RAM/disk admission;0–3/5 foreign workloads were protected. CPU scoring exec92741, refinement94627 and independent verification chunk9687b6 exited0; hostps andNVML showed all owned measurement PIDs absent at23:29:18UTC. [Actual process completion](process_completion.json).

One CPU figure/coverage-helper import failed exit1 due an unterminated f-string before scientific operations. Fixed ordinary syntax, py_compile checked; only CPU metadata/figure/I-O helper retried. One exact-line patch miss made no mutation. Initial duplicate example corrected and selected16/full-grid wording clarified without recomputing statistics or any model/VJP/raster. Original scoring source v1 and figure selection are archived in engineering_revisions, including [failure record](engineering_revisions/cpu_figure_lookup_v2_failure.json). A later CPU-only packaging refresh exited1 because an already updated outline row no longer matched the original exact label; the idempotent prefix match was fixed and packaging retried. This affected no measurements, statistics or figures; its record is engineering_revisions/cpu_packaging_v2_failure.json. Actual measurement code and source identity remain indexed; v1 scorer archive matches its recorded initial hash.

## Actual costs, budgets and limits

Exactly4 CUDA accesses/4DINO loads, **97 DINO forwards** (96gradient +1preflight),96 actual GS rasters,96 layout constructions,6144 VJP started and6144 complete.0training/updates/new native depth/downloads/discarded GPU work. Four GPU worker processes including preflight consumed **{gpu:.6f}s={gpu/3600:.8f}workerh** total, including loading/recovery allowance/I-O/actual wait tails. CPU prepare plus GPU-child wall totals{wall:.6f}s; separate CPU scoring8.30425s and refinement6.82808s body wall include evaluation/figures, imports additionally covered by conservative CPU accounting.

Actual synchronized wall sums: DINO-only load{timers['encoder_load']['sum_seconds']:.6f}s; extra preflight forward{timers['preflight_cached_reference_forward']['sum_seconds']:.6f}s;96gradient forwards{timers['input_gradient_forward']['sum_seconds']:.6f}s;96layout builds{timers['density_layout']['sum_seconds']:.6f}s, **including** nested actual raster{timers['actual_raster']['sum_seconds']:.6f}s;6144joint VJP/energy/transfer/durable-I-O intervals{timers['VJP_complete']['sum_seconds']:.6f}s. The latter is not an isolated backward kernel timer; original pure backward and separate I-O subintervals were not instrumented. Separately recorded [CPU I-O calibration](io_cost_calibration.json) on fixedfirstquery/source measures actual write/fsync/reload of full524416-byte energy files and8192-byte probes; OS cache/exclusivity differ and it cannot be subtracted to fabricate original pure kernel time. [Cost ledger](cost_actual.json) gives exact boundaries, means/min/max and inherited formation scope. New image forward/layout/64VJP are actually paid; saved energy or kernel cache is never called complete native inference. This round did not run native/DPT queries and makes no native latency claim.

Existing J formation is a **dependency**, not a free frontend: its prior report records mean9.879s/place for300-step fitting after initial field/target materialization, with shared source preparation/cold costs separately limited; no fresh formation/end-to-end acquisition timing was run here. Twelve fields are reused for eightimages/place and three layouts, not copied or refit. [Original formation/cost report](../native_feature_field_pe1/report.md).

[Budget evidence](execution_budget.json): GPU{gpu/3600:.8f}≤2workerh; CPU conservative upper{cpu_upper/3600:.6f}≤4workerh, charging4slots continuously from23:13UTC through packaging+120s reserve, including helpers/failure/idlegaps. Conservative admitted host working-set20GiB (3×6+2)≤32. A concurrentRSS snapshot was4,668,710,912bytes; CPU scoring/refinement peakRSS859,533,312bytes. **No fine-grained whole-GPU-phase host peak trace was persisted**, so the4.35GiB snapshot is not presented as the actual peak; the20GiB analytical admission bound and sampled live guard are the resource evidence. No RAM staging, full checkpoint copies, large field copies or new DATA. Own permanent+code bytes before packaging{size:,}; conservative peak with32MiB final/atomic/temporary/doc margin{size+32*2**20:,}bytes≤3GiB. Final manifest gives exact final files. Old results/data and system GLIBC remain unchanged.

Implementation correctness and evidence are complete for this finite measurement. The main G/P condition is unmet; separate positive G/S evidence is conditional, source increment uncertain, coverage and broad sensitivity limit generalization. These are used development data, fixed probes and one DINO/J configuration. No extrapolation to all representations, accurate physical geometry, PE effect, native benefit or fresh confirmation is warranted. Original Astra receives the complete outcome and retains scientific disposition; this implementation does not dispatch a successor or reopen a closed family.
'''
(OUT/'report.md').write_text(report)

short='**round023实施已完成，待原Astra处置（2026-11-14）。** `native_content_support1`完成96图／6144输入VJP，0训练；query G−P +0.8902pp，95%[−0.0548,+2.0968]，未满足预定≥+1pp且下界为正。G−S +1.1026pp [+.2381,+2.2230]单独保留为条件线索；source G−P +.1639pp跨零。查询616/768读可定义，原零覆盖与全部反例保留；测的是单图DINO一阶输入支撑，不代称PE／Native用途或fresh。'
links='[完整报告](../results/gaussian_memory/native_content_support1/report.md) · [唯一合同](../results/gaussian_memory/autonomous_research_loop/round_023/science_plan.md) · [交接](../results/gaussian_memory/native_content_support1/implementation_handoff.json)。round022及全部旧关闭、正负与反例不改；O2–O5沿用，O6只获设计依据，C2性质／C4匹配控制；整体未达、未暂停，无外部阻塞。'
for path in docs:
    lines=path.read_text().splitlines();assert 'round023' in lines[3];lines[3]=short+links;text='\n'.join(lines)+'\n'
    if path.name=='research_outline_alignment.md':
        text=text.replace('只新增实际单图DINO内容的输入支撑测量，尚未实施','已完成实际单图DINO内容的输入支撑测量，主G/P未达预定条件、G/S条件阳性分列，见[报告](../results/gaussian_memory/native_content_support1/report.md)')
        old=next(l for l in text.splitlines() if l.startswith('| [round023实际内容输入支撑'))
        new='| [round023实际内容输入支撑报告](../results/gaussian_memory/native_content_support1/report.md)，实施完成待Astra | O2–O6；C2性质测量／C4控制 | 96图6144VJP；共同读行／同面积；query G−P+.8902pp [−.0548,2.0968]未达条件，G−S+1.1026pp条件阳性；source弱，全部零覆盖／反例及图审阅完成。 | 仅单图DINO一阶支撑，非PE／Native／fresh；原Astra处置，旧家族及round022关闭保持。 |'
        text=text.replace(old,new)
    path.write_text(text)

write(OUT/'CPU_delivery_runtime.json',dict(complete=True,body_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_gradient_raster_calls=0,statistics_recomputed=False))
files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','implementation_handoff.json')]+sources+docs
manifest=dict(scope='All newly owned round023 evidence/code plus exact living-document versions. No old model/field/data rehash. Manifest/handoff excluded to avoid circular identities.',
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))])
write(OUT/'artifact_manifest.json',manifest)
handoff=dict(status='implementation_complete',experiment='native_content_support1',contract=dict(path=str(ROUND/'science_plan.md'),sha256=cfg['science_sha256']),
    report=dict(path=str(OUT/'report.md'),sha256=sha(OUT/'report.md')),manifest=dict(path=str(OUT/'artifact_manifest.json'),sha256=sha(OUT/'artifact_manifest.json')),
    artifacts={n:str(OUT/n) for n in ['summary.json','method_decision.json','execution_budget.json','cost_actual.json','process_completion.json','figure_review.json','configuration.json','input_identities.json','original_zero_coverage.json']},
    results=dict(query_G_P=q['metrics']['CG_CP_pp'],criterion_met=False,query_G_S=q['metrics']['CG_CS_pp'],source_G_P=s['metrics']['CG_CP_pp'],independent_confirmation=False,PE_or_native_utility=False),
    actual_GPU_workers_all_exited=True,all_CPU_science_stages_actual_exit0=True,GPU_worker_hours=gpu/3600,
    unresolved_material_scientific_changes=[],limitations=['Original GPU-phase hostRSS peak not finely persisted;20GiB conservative bound and observed snapshot disclosed.','Joint VJP/reduction/durableI-O wall recorded; pure backward and original separateI-O unavailable, separate CPU calibration disclosed.','Conditional4probe used-development measurement; original Astra scientific disposition outstanding.'],
    original_model_request='gpt-6.1-sol/high',Fast_requested_or_verified=False,backend_effective_unknown=True,successor_dispatched=False)
write(OUT/'implementation_handoff.json',handoff)
print(json.dumps(dict(report_sha256=handoff['report']['sha256'],manifest_sha256=handoff['manifest']['sha256'],handoff_sha256=sha(OUT/'implementation_handoff.json'),GPU_worker_hours=gpu/3600,CPU_upper_hours=cpu_upper/3600,permanent_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())),indent=2))
