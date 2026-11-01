import csv,hashlib,io,json,shutil,time
from pathlib import Path
R=Path('/home/liwt/GRoPE');D=Path(__file__).resolve().parent;O=R/'results/round39_dual_overlap'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
 assert not (D/'before_execution.json').exists()
 shutil.copyfile(__file__,D/'finalize.before.py')
 write(D/'before_execution.json',dict(created_unix=time.time(),source_sha256=sha(__file__),purpose='post-result report and root delivery acceptance',new_array_decodes=False,new_model_calls=False))
 bindings={}
 def bind(p,h=None):
  p=Path(p);digest=sha(p);assert h is None or digest==h,str(p);bindings[str(p.relative_to(R))]=digest
 def complete(p):
  c=read(p/'completion.json');assert c['passed'];bind(p/'completion.json')
  names={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.name!='completion.json'}
  assert names==set(c['files']),str(p)
  for n,h in c['files'].items():bind(p/n,h)
  return c
 receipt=read(O/'prelabel_root_acceptance.json');assert receipt['accepted_for_label_stage'] and receipt['numerical_audit_passed']
 bind(O/'prelabel_root_acceptance.json')
 for n,h in receipt['files'].items():bind(R/n,h)
 primary=complete(O/'evaluation');independent=complete(O/'metrics_audit');controller=complete(R/'results/round39_label_execution')
 assert all(x['exit_code']==0 for x in controller['stages'])
 for s in controller['stages']:bind(R/s['log'],s['log_sha256'])
 s=read(O/'evaluation/summary.json');gate=read(O/'evaluation/screen.json');audit=read(O/'metrics_audit/verification.json')
 assert audit['passed'] and gate==read(O/'metrics_audit/screen.json')
 passed=sum(x['passed'] for x in gate['checks']);assert len(gate['checks'])==8
 old=read(R/'results/round20_anchor_fixed/summary.json');bind(R/'results/round20_anchor_fixed/summary.json')
 quad=old['old_by_role']['strong_quad']['by_condition']['g24_g8']['kl']
 report=R/'docs/round39_dual_overlap_report.md';assert not report.exists()
 table='\n'.join(f"| {role} | {s[role]['region']['macro']['kl']:.9f} | {100*s[role]['center']['macro']['pck']['value']:.6f}% | {s[role]['center']['macro']['rmse']:.6f} |" for role in ['full','isotropic','point'])
 per=[]
 for scene,row in s['full']['region']['by_scene'].items():
  c=s['full']['center']['by_scene'][scene]
  per.append(f"| {scene} | {row['kl']:.9f} | {c['pck_count']}/{c['n_point']} = {100*c['pck']['value']:.6f}% | {c['rmse']:.6f} |")
 checks='\n'.join(f"| {c['reference']} | {c['metric']} | {(c['bound']['value'] if isinstance(c['bound'],dict) else c['bound']):.9f} | {'通过' if c['passed'] else '失败'} |" for c in gate['checks'])
 num=read(O/'audit/checks.json');fwd=read(O/'forward/completion.json');nc=read(O/'audit/completion.json')
 report.write_text(f'''# R39：双侧三维 Gaussian overlap 未通过旧 TRAIN 必要筛查

2026-11-01。固定主方案 **{passed}/8 项通过，停止该固定设计**。2304 个查询、全部三臂预测及独立数值验证完整，两个标签指标实现一致。没有触及 VAL、实际 V、真实 GS 或新的确认集；没有修改原门槛，也没有将消融臂替换成新主方案。总体研究目标仍未完成。

## 检验的唯一设计

每个原生 patch 只保留一个中心 a 和三维二阶矩 Q；中心取预测中心射线，Q=mean(pixel covariance)+mean[(x−a)(x−a)ᵀ]。双方都从原 RGB 预测深度构建，query 变换到 key 相机坐标。采用已知 Gaussian kernel-mean overlap：

    S = γI + Qq + Qk
    log K = log(mq mk) + (3 log γ − log det S − δᵀ S⁻¹δ)/2
    P = softmax(log(K + mq mk × 1e−30) + 原内容偏置)

γ 固定为两完整图像典型像素世界长度平方的 4 倍，mq=9、mk=1。没有训练新模块、扫描带宽、改温度或增加另一个中心读出。full、point、isotropic 获得完全相同的两侧深度、中心、γ、内容及候选集合；后两者分别移除支撑、保留同迹但移除各向异性。

额外的 key 深度是本次明示的信息变化，不是协方差自己的收益。这个核并不新颖，已有 [核均值闭式来源](https://papers.neurips.cc/paper_files/paper/2012/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper.pdf)；本仓库也已有二维 overlap 和另一种三维 Bhattacharyya 核。按新网格重拟合单 Gaussian 不保持精确划分可加性，本轮没有借用 R38 的积分-law 数值证明。

## 结果

全部 48 对、三场景各16对；每对48个q24查询、432个g8 key。原1096个真值有效中心、原像素可见质量和等场景宏平均均保留。区域 KL 越低越好；PCK@8 越高越好。

| 固定角色 | 宏 KL | 宏 PCK@8 | 宏 RMSE（px） |
|---|---:|---:|---:|
{table}

主方案 KL={s['full']['region']['macro']['kl']:.9f}，原 strong_quad/legacy_quad16 的共同参考为 {quad:.9f}；劣化 {100*(s['full']['region']['macro']['kl']/quad-1):.3f}%，超过原允许2%的上限。两个名字对应同一个配置，不能算成两份独立证据。

同信息 full 相对 point 的区域分布更好，但 full 的 PCK 仍较低；isotropic 的 KL 比 full 更好、PCK 更差。这个结果不支持在本设置下同时取得区域与定位优势，更不支持越过旧二维强对照。不能根据这些消融差异直接认定深度不一致、determinant 偏好或内容强度就是原因。

| 主方案场景 | KL | PCK@8（原计数） | RMSE（px） |
|---|---:|---:|---:|
{chr(10).join(per)}

| 原参考 | 指标 | 原通过边界 | 判定 |
|---|---|---:|---|
{checks}

## 数值与证据边界

事前快照的10项合成测试全部通过，包括直接 Gaussian 期望的20/32阶三维 Gauss–Hermite 交叉验证、常深度平面、Dirac/PSD、共同刚体与尺度变换、depth交换及极端floor。真实数据的144项独立矩阵检查覆盖48对×3臂，全部通过；这里复算的是新闭式算子，未重跑旧R38参考。

最大几何归一化误差 {max(x['geometry_error'] for x in num):.5g}，最大log-P归一化误差 {max(x['score_error']['logp'] for x in num):.5g}，最大P绝对误差 {max(x['probability_error'] for x in num):.5g}；所有原生first-index winner精确一致。原q像素几何重建误差为零。生产阶段 {fwd['phase_wall_seconds']:.3f}s，独立数值阶段 {nc['phase_wall_seconds']:.3f}s；这包含来源核验与保存，不是原正式完整算子速度门，也不能与R38单查询耗时直接计算加速比。三臂全部数组共 {fwd['forward_npz_bytes']:,} bytes。

源代码/协议在合成测试前及正式执行前封存，独立数值核验和根验收先于标签读取。双标签路径最大逐行差 {audit['maximum_discrepancy']['rows']:.5g}、汇总差 {audit['maximum_discrepancy']['summary']:.5g}，八项判定完全相同。所有阶段exit0，未重试、未修改冻结源、未省略失败查询。

## 后续约束

停止当前γ和双侧支撑定义，保留其失败。下一步先提出可证伪的机制诊断，区分双视图深度对齐、log determinant偏好及支撑形状各自的影响；目前尚无证据选定其中一个解释，也未选择新的修补参数。原TRAIN全条件、VAL/扰动/完整成本、实际V、真实GS和未用数据联合确认门全部仍在。一个结构简洁、实现正确的公式不等于有效方案。

[固定协议](../ROUND39_DUAL_OVERLAP_PROTOCOL.md) · [三臂汇总](../results/round39_dual_overlap/evaluation/summary.json) · [原八门](../results/round39_dual_overlap/evaluation/screen.json) · [独立数值检查](../results/round39_dual_overlap/audit/checks.json) · [独立指标核验](../results/round39_dual_overlap/metrics_audit/verification.json) · [根验收](../results/round39_dual_overlap_root_acceptance_20261101.json)。
''')
 bind(report)
 for name in ['finalize.py','finalize.before.py','before_execution.json']:bind(D/name)
 rootreceipt=R/'results/round39_dual_overlap_root_acceptance_20261101.json';assert not rootreceipt.exists()
 write(rootreceipt,dict(created_unix=time.time(),accepted_for_development_failure_interpretation=True,numerical_audit_passed=True,dual_metric_routes_agree=True,passed_checks=passed,total_checks=8,primary='full',regional_kl=s['full']['region']['macro']['kl'],pck=s['full']['center']['macro']['pck'],decision=primary['decision'],source_unchanged=True,labels_opened_after_root_acceptance=True,scientific_gate_passed=False,goal_complete=False,files=bindings))
 write(D/'completion.json',dict(passed=True,root_acceptance_sha256=sha(rootreceipt),report_sha256=sha(report),files={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()},new_array_decodes=False,new_model_calls=False))
 print('R39_ACCEPTED_FAILURE',passed,8,'bindings',len(bindings),'root_sha256',sha(rootreceipt),flush=True)
if __name__=='__main__':main()
