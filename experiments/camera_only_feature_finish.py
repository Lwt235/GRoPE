"""Final immutable delivery identities and concise living-document synchronization."""
import hashlib
import json
from pathlib import Path
from experiments import camera_only_feature_data as d

def identity(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha(p))

def main():
    assert d.read(d.OUT/'figures/visual_review.json')['actual_view_image_reviewed_both']
    validation=d.read(d.OUT/'validation_complete.json');assert validation['complete']
    config=d.read(d.OUT/'configuration_lock.json');revision=d.read(d.OUT/'runtime_source_revision_v2.json')
    for name,h in config['science_inputs'].items():assert d.sha(d.SCIENCE/name)==h
    assert d.sha(d.ROOT/'docs/camera_only_feature_pe_execution_plan.md')==config['execution_sha256']
    for path,h in revision['current_source_sha256'].items():assert d.sha(d.ROOT/path)==h
    inputs=d.read(d.OUT/'input_manifest.json')
    for name,r in inputs['inputs'].items():
        st=Path(name).stat();assert st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns'],name
    budgets=d.read(d.OUT/'budget.json')['counts']
    assert budgets==dict(formal_updates=18000,pilot_updates=6,training_reader=36000,training_source=36000,pilot_reader=12,pilot_source=12,native=4570,reader=3441,source=590,raster=307),budgets
    s=d.read(d.OUT/'scoring/summary.json');flags=d.read(d.OUT/'method_decision.json');p=s['primary'];pe=s['comparisons'][p+'/G-P'];cost=d.read(d.OUT/'cost.json');fmt=lambda r:f"{r['reduction_percent']:+.4f}% [{r['interval95_percent'][0]:+.4f}, {r['interval95_percent'][1]:+.4f}]%"
    backup=d.OUT/'documentation_before';backup.mkdir(exist_ok=True);updates={}
    opening=f'''**round006实施完成（2026-11-13，camera_only_feature_pe1）：完整结果交原Astra评价。** 九次各2000更新、396正常＋36置换公共状态、3312feature／4368native预测、独立NumPy复算、固定配对bootstrap、同调用成本及两图实际审阅齐备。开发原query主空间G/P降误差{fmt(pe)}，预定PE筛flag={flags['PE_empirical_flags']['G-P']['practical_screen']}；G/S={fmt(s['comparisons'][p+'/G-S'])}，S/P={fmt(s['comparisons'][p+'/S-P'])}。三组对Native净用途、RasterFill材料性代价及全部辅助／训练／其他query结果均保留，不以feature替换主端点，不把共同读取收益归为PE。未选择候选、未宣布等价／非劣性或总体成功。[完整报告](../results/gaussian_memory/camera_only_feature_pe1/report.md) · [全数表](../results/gaussian_memory/camera_only_feature_pe1/report_tables.md) · [实测flags](../results/gaussian_memory/camera_only_feature_pe1/method_decision.json)。

本轮仅小camera-only reader学习完整encoder feature残差，VGGT全部冻结；P/S/G共享完整Σ普通内容、中心邻域／密度、真实render anchor和相机，以同29维位置预算比较K位置组织，非nested独占信息。源K/V一次保存重载供8query读取，原GS保留只读依赖；10个原覆盖全零query及最大反例均未筛除。O2–O5公共GS／读取／渲染与O6 memory reader PE分列，C1/C5保持、C2/C3/C4/C6按计划改变；给定相机尺度、sensor/joint-validity、静态四源、2048×1024、固定场、已知ROI／depth／旧数据仍为临时范围。历史区域正证据、Render／writer条件用途及近轮阴性继续有效，无verified fresh reserve。

实际GPU含失败／加载／I/O为{cost['GPU_worker_hours']:.6f}worker小时≤6，全部自有GPU子进程正常退出；0native训练，六discarded pilot不重复，失败环境／恢复断言仅作原轮版本修复。实际预算native4570／4650、reader3441／3500、独立source590／600、真实raster307／400；120次reader成本包含在200次完整query内，无额外benchmark。DATA约{cost['DATA_GiB']:.4f}GiB≤18，项目≤1GiB，恢复与资源账完整。主复用原`/root/research_analysis_astra`作取舍，不自动后继；本CLI turn结束释放实施租约。实现UUID仍`01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，请求Sol/high/fast，主client观察与后台实际值分列，tier unknown；未操作主goal/state/events或全局认证。

'''
    for name in ['research_current.md','research_outline_alignment.md','gaussian_memory_evidence_scope.md']:
        path=d.ROOT/'docs'/name;old=path.read_bytes();saved=backup/name
        if not saved.exists():saved.write_bytes(old)
        else:old=saved.read_bytes()
        head,rest=old.decode().split('\n',1);mark=rest.index('<details>');prior=rest[:mark]
        path.write_text(head+'\n\n'+opening+'<details>\n<summary>round006事前科学计划／派发记录（现已完成实施，保留形成时文字）</summary>\n'+prior+'\n</details>\n\n'+rest[mark:])
        updates[str(path)]=dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=d.sha(path),before_bytes=len(old),after_bytes=path.stat().st_size,positive_growth=max(0,path.stat().st_size-len(old)))
    d.write(d.OUT/'documentation_updates.json',updates)
    # Data identities are already hashed at atomic publication and independently rechecked
    # while scoring. Reuse those completion hashes, not a second14GiB read.
    data={}
    for marker in (d.OUT/'completed').glob('*.json'):
        r=d.read(marker)
        for kind in ['feature','depth']:
            path=Path(r[kind+'_path']);data[str(path)]=dict(bytes=path.stat().st_size,sha256=r[kind+'_sha256'],authority=str(marker),independent_scoring_verified=True)
    for marker in (d.OUT/'baselines').glob('*.json'):
        r=d.read(marker)
        for entry in r['methods'].values():
            path=Path(entry['path']);data[str(path)]=dict(bytes=path.stat().st_size,sha256=entry['sha256'],authority=str(marker),independent_scoring_verified=True)
    for marker in (d.OUT/'states').glob('*.json'):
        r=d.read(marker);path=Path(r['path']);data[str(path)]=dict(bytes=path.stat().st_size,sha256=r['sha256'],authority=str(marker),public_base_reference=r['base_reference'],base_sha256=r['base_sha256'])
    for path in d.CACHE.rglob('*'):
        if path.is_file() and str(path) not in data:data[str(path)]=dict(bytes=path.stat().st_size,sha256=d.sha(path))
    source_names=['models/camera_only_feature_reader.py','experiments/camera_only_feature_data.py','experiments/camera_only_feature_run.py','experiments/camera_only_feature_launch.py','experiments/camera_only_feature_score.py','experiments/camera_only_feature_delivery.py','experiments/camera_only_feature_finish.py','experiments/camera_only_feature_load_cost.py']
    artifacts={}
    for path in d.OUT.rglob('*'):
        if path.is_file() and path.name not in ('manifest.json','implementation_handoff.json') and not (path.name.startswith('finish') and path.suffix=='.log'):artifacts[str(path)]=dict(bytes=path.stat().st_size,sha256=d.sha(path))
    manifest=dict(complete=True,created_utc=d.now(),scientific_input_identities=config['science_inputs'],execution_sha256=config['execution_sha256'],source_sha256={name:d.sha(d.ROOT/name) for name in source_names},project_artifacts=artifacts,data_artifacts=data,documentation_updates=updates,original_inputs=inputs,hash_policy='New prediction publication SHA independently checked in scoring; reuse here. No old large weight hashing. Manifest/handoff self excluded.')
    d.write(d.OUT/'manifest.json',manifest)
    project_growth=d.diskbytes(d.OUT)+sum(r['positive_growth'] for r in updates.values())+sum((d.ROOT/name).stat().st_size for name in source_names)
    assert project_growth+2**20<2**30
    outputs={name:identity(d.OUT/name) for name in ['report.md','report_tables.md','method_decision.json','scoring/metrics.json','scoring/summary.json','scoring/independent_recomputation.json','cost.json','cost_amortized_totals.json','budget.json','configuration_lock.json','input_manifest.json','runtime_source_revision_v2.json','validation_complete.json','figures/outcomes.png','figures/cases.png','figures/visual_review.json','manifest.json','public_progress.jsonl']}
    handoff=dict(schema_version=1,round=6,experiment='camera_only_feature_pe1',status='complete',completed_utc=d.now(),interface='existing Codex CLI exact UUID resume',CLI_UUID='01a1d350-d1ec-70f2-9e37-8960b2dfe7a8',requested_configuration=dict(model='gpt-6.1-sol',reasoning_effort='high',service_tier='fast',source='Actual parent CLI invocation described by human dispatch'),client_configuration=dict(model='gpt-6.1-sol',effort='high',approvals_reviewer='auto_review',source='Parent rollout observation reported in dispatch, not role independent backend attestation'),effective_backend_configuration=dict(model='unknown',effort='unknown',service_tier='unknown',independently_verified=False),inputs=config['science_inputs'],execution_document_sha256=config['execution_sha256'],original_scientific_plan_unchanged=True,outputs=outputs,DATA_root=str(d.CACHE),fits=9,formal_updates=18000,formal_reader_source_samples_each=36000,discarded_pilot_updates=6,pilot_reader_source_samples_each=12,normal_states=396,permuted_states=36,feature_predictions=3312,native_predictions=4368,primary_endpoint=p,primary_PE_contrasts={pair:s['comparisons'][p+'/'+pair] for pair in ['G-P','G-S','S-P']},empirical_flags=flags,budget_actual=budgets,budget_limits=d.LIMITS,GPU_worker_hours=cost['GPU_worker_hours'],DATA_bytes=d.diskbytes(d.CACHE),project_new_bytes_upper_bound=project_growth+2**20,ordinary_fixes=['Ninja and old extension-cache PATH reused;1failed raster counted','CPU/CUDA optimizer step comparison fixed; Pstep1 checkpoint resumed without repeating consumed pilot'],all_native_weights_frozen=True,native_training_forwards=0,public_state_scope='432 derived K/V states require44 immutable original GS fields as references; not self-contained single files. Same normal state reused8queries.',no_query_RGB_feature_depth_mask_ID_in_reader=True,full_condition_scope='Native_full common by construction, not learned fidelity/noninferiority',fresh_confirmation=False,verified_fresh_reserve=False,general_research_goal_achieved=False,scientific_candidate_selected=False,all_GPU_processes_exited=True,CLI_actual_turn_completion='Coordinator observes after this final response; not pre-attested',coordinator_goal_tools_called=0,coordinator_state_events_edited=False,agents_or_tasks_created=0,cross_thread_messages_sent=0,old_or_foreign_processes_or_results_modified=False,global_auth_or_config_modified=False,scientific_write_lease='Released at end of this actual CLI turn; no next round started',remaining_this_finite_stage=[],next_required_input='Coordinator checks complete artifacts then reuses original /root/research_analysis_astra for interpretation and scientific choice; no successor chosen by implementation.')
    d.write(d.OUT/'implementation_handoff.json',handoff)
    print(json.dumps(dict(complete=True,report_sha256=d.sha(d.OUT/'report.md'),handoff_sha256=d.sha(d.OUT/'implementation_handoff.json'),manifest_sha256=d.sha(d.OUT/'manifest.json'),project_new_bytes_upper_bound=handoff['project_new_bytes_upper_bound'])))

if __name__=='__main__':main()
