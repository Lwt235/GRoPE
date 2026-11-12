# rendered_feature_native1 active finite goal

User accepted docs/learned_projection_results_analysis_20261112.md by “设为目标并执行”. Formal goal is ACTIVE; do not mark complete until all fixed outputs, review, report and docs are done. No automatic scientific successor.

Read docs/rendered_feature_native_execution_plan.md and current/outline. 9 fits Local/Global/Render × seeds20261110/11/12, 2000 updates each. 864 adapted +288 valid-slot shuffle native depth exports, independent FP64 rescoring, same primary missing ROI spatial loss, actual two-figure review and fresh query cost.

All11 new modules compiled. Model/fusion: models/gaussian_scene_rendered.py. New experiments prefix rendered_feature_. Base scientific files gaussian_scene_memory/native/train/config unchanged. Common787712-parameter2048→256→1024 GELU residual, zero output init, once at encoder output, original24layers/DPT frozen. Local=current patch mean, Global=effective alpha*saved valid feature mean, Render=sum actual weights*saved valid content. Derived FP32 features from existing sensor weights; source fields unchanged. Preflight6cases: native maxdiff0, actual direct renderer vs cache max2.861e-6, peak5.621GiB. All24field/four-query caches done. Data cache outside repo at DATA/rendered_feature_native_v1 (authorized host writes).

Host CPU orchestrator22593, logfile logs/orchestrator.log; postprocessor29529, logfile logs/postprocess.log, waits12exports. Verify live exact /proc argv before restart; default sandbox /proc/CUDA are not real host, use escalation. Queue automatically schedules idle GPUs or at most two foreign-shared cards;7GiB cap,13GiB free admission, own worker yields below6GiB free, max4workers, one/card. First Local/Global use7/6. Do not touch foreign jobs. Old training loops and checkpoint/sample matching reused by process-local aliases only.

Monitor queue_status.json, */progress.json and logs. Resource return75 is requeue, not scientific failure. Native model loads about10–20sec; busy shared cards roughly1.3sec/update observed. Do not evaluate partial scores to select configs or checkpoints. If interim descriptions are needed label them nonfinal. Model/source identities and matched sample checks run after all nine.

Scoring/figures/checks/finish/cost modules installed. Cost queued after first seed dependencies, may overlap later fits, fresh prepared RGB→encoder→actual renderer→single fusion→all native/DPT,12cases2warmup+5retained. Shared timing cannot support exclusive speed ranking. Source construction is unchanged inherited sensor cost, not RGB end-to-end cost. Inherited metrics parent_L/native_current and previous_P/S/G are clearly background, not matched new controls or fresh evidence.

After fits: inspect postprocess output, score summary, all auxiliary costs, actual view outcomes.png and depth_cases.png (fixed3+largest R−Local regression). Then record actual SHA/review notes, write complete scientific report and method_decision.json, synchronize current/outline/execution/analysis/evidence scope as needed. Distinguish O4/O5 memory utility from unresolved O6 PE; never call direct render utility PE proof. Complete finite goal only after required work. Keep negative results and temporary sensor/given-camera/static/frozen/old-data limitations. No more seeds/formulas/modules by default.

## First two fits complete

Local_20261110 and Global_20261110 each finished2000steps and96exported predictions, all files/shape/finite identities checked. Actual first-seed Local/Global/Render initial tensors and sample schedules identical; first_seed_initialization.json. No scoring used to change plan. Queue now Render_20261110 and Local_20261111 onGPU7/6. Initial observation briefly showed one task during CUDA cleanup; queue restored two by itself, no admission policy/source/config changes. A host-read automatic approval timed out once; allowed one retry succeeded, no training failure or restart. /tmp/rendered_feature_watch.py is a reusable stdlib observer; run with escalated host access for real /proc identity, not default sandbox. It updates dated execution_status only.

## First paired seed and4of9fits complete

First Local/Global/Render and second Local completed2000steps. Partial scoring collected480new exports (384adapted+96shuffle), max independent FP64 difference1.31839e−15; source partial_scoring/metrics.json and independent_recomputation.json. first_seed_description.json slices SEEDS[0] only: main Local/Global/Render .025018411/.027105624/.025586679; R/Local −2.2714% error reduction, R/Global +5.6038%; correct vs shuffle +25.9298%. Raw missing R/Local −19.5212%, full wholeAbsRel −15.4826%. NoCI or final method decision; rest unchanged. SecondGlobal/Render nowGPU1/7 launchers53403/53995. Prior secondLocal launcher was briefly zombie in stale queue snapshot after successful export; actual terminal log confirmed96exports and dispatcher advanced normally. Do not restart from transient observation. Three living docs updated with nonfinal scope. /tmp watcher now derives completed_fits and current export counts from queue/markers to avoid stale snapshot fields.

## Interpretation-only code check while fits continue

interpretation_notes.md records that current fusion receives image features plus rendered content, not explicit3Dcenters/metre depth/source-baseline channels. Geometric parameters enter weights only. An ideal pinhole one-query scaling thought example describes a representation limit, not a new experiment or proved actual failure cause. surface1 and geometry_transport1 already tested geometry/depth in K/V or direct output and had conditional/negative results; do not propose adding depth as a first/new guaranteed fix. This note changes no configuration, loss, scope, budget or successor. Use it only as bounded interpretation in final report if relevant.

## Resource yield observed

Global_20261111 startedGPU1 (free36GiB then), external job later jumped memory, free170MiB observed. Launcher stopped only own CUDA child53499 and returned75; unchanged queue requeued/resumedGPU6 fromlatest300. New launcher60535; retained logs contiguous, elapsed timer reset once, interrupted_after_300.jsonl preserves discarded tail. No source/config/precision/schedule/budget edits. Resource resume_description.json and execution plan note added. Final timing summary must count discarded work separately, as existing code does. Do not call this a scientific failure or add a new gate.

## Two paired seeds /6of9fits complete

SecondGlobal/Render finished; partial_scoring all768new predictions independently scored, max1.31839e−15. two_seed_description.json noCI: Local/Global/Render main .024985578/.027384061/.025645877; R/Local reduction−2.64272% (−2.2714/−3.0150 seeds), R/Global+6.34743% (+5.6038/+7.0761), R/cf+26.09412%. Missing raw R/Local−16.86449%, fullwholeAbsRel−15.98282%. No final decision or tuning. ThirdLocal/Global running, thirdRender still queued. Partial training_summary.json completed6/pending3, final finish will replace it after all9. Current/outline/execution updated. Interpretation notes add a key scope: Local secondinput is current global mean, R secondinput rendered source context, so alternative context operators at same capacity, not nested completeLocal+ignorable source channel. Do not call R loss proof all source conditional info is zero. No source/code/plan changes.

## Eight fits and query cost complete

ThirdGlobal/Local finished2000steps and96exports each.960new exports available (8adapted*96+2shuffle*96); only768previously independently rescored at this point, final finish must rescore all1152. FinalRender_20261112 launchedGPU7, launcher61260, active. Query_cost completed onGPU6 while finalR trains. cost.json12cases2warmup/5retained, fresh actual renderer+encoder+native. cost_description.json complete with raw/code/sourceSHA and scope: L/U/R combined20samplemedian273.6695/305.7299/300.6823ms; ranges large shared, no speed ranking/cross-round improvement. First process model load16.54896s; peak5.20429GiB includes all3fusers/field. Scene4.11954MiB, devcache1.12853GiB (actual exact inJSON), no deployed-cache conflation. Partial training_summary.json now8complete/1pending; last fit pending. Current/outline/execution synchronized. No final effect or completion yet.


## 最终完成（2026-11-12）

九次2000步、864适配及288绑定打乱输出完成；1152份新预测独立复算最大差1.67e−15，成本及两图实际审阅完成，七份研究文档同步。真实主机精确模块/UID/CWD检查无本轮运行进程，completion_audit.json全部有限交付成立、无剩余工作。

R对Local主空间误差高3.20%，三种子均退步且区间宽；原始缺失/完整全图AbsRel高18.99%/17.15%，不采用Render/Global，不晋升Local。绑定依赖25.77%和支持ROI对Global的13.60%改善保留。旧数据、传感器共同有效性、静态给定相机/冻结与非严格嵌套控制边界保留；O6未检验，总体有效方法目标未达成。不自动启动后继研究。
