"""Forward-only diagnostic of replacing actual key cells by Gaussians.

No labels are opened. Query supports stay fixed while genuine key partitions
are split/merged. Independent NumPy Gaussian overlap and SciPy rectangle mass
references distinguish algebraic additivity from task accuracy. The actual
positive cell-integral implementation is checked separately on the same inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import _mvn
import torch

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ('g8', 'g16', 'mixed')
WIDTHS = (.25, .5, 1., 2.)
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def query_moments(inputs, owner):
    xy = inputs['q_to_k_xy'].astype(np.float64)/8
    pixel_cov = inputs['q_to_k_pixel_cov'].astype(np.float64)/64
    means, covariance = [], []
    for i in range(108):
        selected = owner.ravel()==i
        m = np.mean(xy[selected], axis=0)
        delta = xy[selected]-m
        means.append(m)
        covariance.append(delta.T@delta/selected.sum()+pixel_cov[selected].mean(0))
    return np.asarray(means), np.asarray(covariance)


def overlap_mass(mean, covariance, key_mean, key_cov, key_area, width):
    """Area-scaled Gaussian density in common8 units, independently NumPy."""
    pair_cov = covariance[:,None]+key_cov[None]+width**2*np.eye(2)
    delta = mean[:,None]-key_mean[None]
    sign, logdet = np.linalg.slogdet(pair_cov)
    if not np.all(sign>0): raise ValueError('pair covariance not SPD')
    inv = np.linalg.inv(pair_cov)
    distance = np.einsum('nki,nkij,nkj->nk',delta,inv,delta)
    # area/64 is needed for physical raw mass; the /64 cancels under softmax.
    return np.exp(-.5*(distance+logdet))/(2*np.pi)*(key_area[None]/64)


def scipy_rectangle_mass(mean, covariance, bounds, width):
    output = np.zeros((len(mean),len(bounds)), np.float64)
    diagnostics = dict(negative_count=0, minimum_before_roundoff_clip=0.,nonzero_status_count=0)
    for i,(m,c) in enumerate(zip(mean,covariance)):
        for j,b in enumerate(bounds):
            value,status = _mvn.mvnun(b[:2],b[2:],m,c+width**2*np.eye(2),100000,1e-13)
            diagnostics['nonzero_status_count'] += int(status!=0)
            diagnostics['negative_count'] += int(value<0)
            diagnostics['minimum_before_roundoff_clip'] = min(diagnostics['minimum_before_roundoff_clip'],value)
            if value < -1e-12 or status:
                raise ValueError('SciPy reference failed stated absolute tolerance')
            output[i,j] = max(0.,value)
    return output, diagnostics


def collapse(mass, parent):
    output = np.zeros((len(mass),108),np.float64)
    for key,p in enumerate(parent):output[:,p]+=mass[:,key]
    return output


def comparison(a,b):
    raw_l1 = np.abs(a-b).sum(-1)
    denom_a,denom_b = a.sum(-1),b.sum(-1)
    usable = (denom_a>1e-12)&(denom_b>1e-12)
    pa,pb = np.zeros_like(a),np.zeros_like(b)
    pa[usable] = a[usable]/denom_a[usable,None]
    pb[usable] = b[usable]/denom_b[usable,None]
    middle=(pa+pb)/2
    def contribution(p):
        return np.where(p>0,p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(middle,1e-300))),0).sum(-1)
    js=(contribution(pa)+contribution(pb))*.5
    l1=np.abs(pa-pb).sum(-1)
    raw_relative=raw_l1/np.maximum(denom_b,1e-300)
    record=dict(queries=len(a),normalization_usable_count=int(usable.sum()),
        normalization_policy='report normalized errors only where both raw FOV masses>1e-12; raw errors retain all queries',
        raw_max_abs_cell_error=float(np.max(abs(a-b))),raw_l1_mean=float(raw_l1.mean()),
        raw_l1_max=float(raw_l1.max()),raw_relative_l1_mean_usable=float(raw_relative[usable].mean()) if usable.any() else None,
        normalized_l1_mean=float(l1[usable].mean()) if usable.any() else None,
        normalized_l1_max=float(l1[usable].max()) if usable.any() else None,
        normalized_js_mean=float(js[usable].mean()) if usable.any() else None,
        normalized_js_max=float(js[usable].max()) if usable.any() else None,
        fov_mass_a_range=[float(denom_a.min()),float(denom_a.max())],
        fov_mass_b_range=[float(denom_b.min()),float(denom_b.max())])
    return record,dict(raw_l1=raw_l1,raw_relative_l1=raw_relative,
                       normalized_l1=l1,normalized_js=js,normalized_usable=usable)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--doc',type=Path,default=ROOT/'docs/round10_key_partition_diagnostic.md')
    args=parser.parse_args()
    if args.out.exists() or args.doc.exists():raise ValueError('refuse overwrite')
    torch.set_num_threads(4)
    # The candidate B operator is used only as a separate numerical check.
    from models.round10_cell_integral import gaussian_cell_mass
    files=[Path(__file__),ROOT/'models/round10_cell_integral.py',args.data/'completion.json',
           args.data/'contract.json',args.data/'partitions.npz']
    source_hashes={str(p.resolve()):sha(p) for p in files}
    completion=json.loads((args.data/'completion.json').read_text())
    contract=json.loads((args.data/'contract.json').read_text())
    if sha(args.data/'contract.json')!=completion['contract_sha256']:
        raise ValueError('changed contract')
    if sha(args.data/'partitions.npz')!=contract['partitions_sha256']:
        raise ValueError('changed partitions')
    parts=read_npz(args.data/'partitions.npz')
    entries=[]
    for scene in SCENES:
        candidates=[entry for entry in completion['manifests'] if entry['scene']==scene and entry['split']=='train']
        entry=candidates[0]
        manifest_path=Path(entry['path'])
        if sha(manifest_path)!=entry['sha256']:raise ValueError('changed pair manifest')
        manifest=json.loads(manifest_path.read_text())
        if manifest['pair_ordinal']!=0 or manifest['split']!='train':raise ValueError('wrong first calibration pair')
        input_path=manifest_path.parent/'inputs.npz'
        if sha(input_path)!=manifest['input_sha256']:raise ValueError('changed observations')
        # Manifest contains provenance, not raw labels; labels.npz is never read.
        entries.append(dict(scene=scene,pair=manifest['pair'],pair_ordinal=manifest['pair_ordinal'],
            manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),
            input_path=str(input_path),input_sha256=sha(input_path)))
    args.out.mkdir(parents=True)
    write(args.out/'lock.json',dict(source_hashes=source_hashes,entries=entries,
        widths=list(WIDTHS),fixed_query_partition='g16',key_partitions=list(PARTITIONS),
        aggregation='sum every key partition into g16 native parent cells',
        label_archives_read=False,content_temperature=0,normalization_floor_for_reporting=1e-12,
        scipy_reference='scipy.stats._mvn.mvnun, maxpts100000, absolute tolerance1e-13; negative roundoff<1e-12 clipped0',
        actual_B='gaussian_cell_mass float64 CPU order24, defaultpositiveconditionalquadrature',
        claim='operator mechanism only; no task accuracy or speed gate',created_unix=time.time()))
    started=time.time()
    scenarios=[]
    for entry in entries:
        inputs=read_npz(entry['input_path'])
        mean,cov=query_moments(inputs,parts['g16_pixel_index'])
        scenarios.append((entry['scene'].replace('/','__'),mean,cov,entry))
    # All 108 constant-depth plane query supports under identical cameras.
    scenarios.append(('analytic_constant_plane_identity',parts['g16_centers']/8,
                      parts['g16_covariance']/64,dict(type='analytic',sensor_or_real_target_used=False)))
    rows=[]; all_checks=[]
    for name,mean,cov,metadata in scenarios:
        arrays=dict(query_mean=mean,query_covariance=cov)
        for width in WIDTHS:
            raw={m:{} for m in ('overlap','scipy_cell','actual_cell')}
            for partition in PARTITIONS:
                bounds=(parts[partition+'_bounds']-.5)/8
                parent=parts[partition+'_parent_g16']
                gauss=overlap_mass(mean,cov,parts[partition+'_centers']/8,
                    parts[partition+'_covariance']/64,parts[partition+'_area'],width)
                scipy,reference_diag=scipy_rectangle_mass(mean,cov,bounds,width)
                with torch.no_grad():
                    actual=gaussian_cell_mass(torch.from_numpy(mean),torch.from_numpy(cov),
                        torch.from_numpy(bounds),bandwidth=width).numpy()
                check=dict(scenario=name,h=width,partition=partition,
                    scipy_diagnostics=reference_diag,max_absolute_actual_vs_scipy=float(np.max(abs(actual-scipy))))
                if check['max_absolute_actual_vs_scipy']>3e-10:
                    raise ValueError('positive rectangle operator differs from independent reference')
                all_checks.append(check)
                for method,value in [('overlap',gauss),('scipy_cell',scipy),('actual_cell',actual)]:
                    raw[method][partition]=collapse(value,parent)
                    arrays[f'{method}_h{width}_{partition}_native_raw']=value
                    arrays[f'{method}_h{width}_{partition}_parent_raw']=raw[method][partition]
                    denominator=value.sum(-1)
                    arrays[f'{method}_h{width}_{partition}_native_normalized']=np.divide(value,
                        denominator[:,None],out=np.zeros_like(value),where=denominator[:,None]>0)
            for method in raw:
                for partition in ('g8','mixed'):
                    record,detail=comparison(raw[method][partition],raw[method]['g16'])
                    rows.append(dict(scenario=name,h=width,method=method,
                        partition=partition,reference_partition='g16',**record))
                    arrays.update({f'{method}_h{width}_{partition}_vs_g16_'+key:value for key,value in detail.items()})
        np.savez_compressed(args.out/(name+'.npz'),**arrays)
        print('DIAGNOSED',name,flush=True)
    summaries={}
    for method in ('overlap','scipy_cell','actual_cell'):
        selected=[r for r in rows if r['method']==method]
        real=[r for r in selected if not r['scenario'].startswith('analytic')]
        summaries[method]=dict(real_mean_normalized_l1=float(np.mean([r['normalized_l1_mean'] for r in real])),
            real_max_normalized_l1=max(r['normalized_l1_max'] for r in real),
            real_mean_normalized_js=float(np.mean([r['normalized_js_mean'] for r in real])),
            max_raw_cell_split_defect=max(r['raw_max_abs_cell_error'] for r in selected),
            all_normalized_queries_usable=all(r['normalization_usable_count']==108 for r in selected))
    result=dict(summary=summaries,rows=rows,numerical_checks=all_checks,seconds=time.time()-started,
        label_archives_read=False,task_effectiveness_established=False,
        mechanism='query Gaussian held fixed; actual rectangle integration is additive under key partition, Gaussian surrogate is not')
    if any(sha(path)!=digest for path,digest in source_hashes.items()):raise ValueError('source changed during diagnostic')
    write(args.out/'summary.json',result)
    body=['# Round10 key partition mechanism diagnostic','',
        '这是无标签的算子诊断，不是任务效果验证。固定每个 g16 query Gaussian，只改变 key 的真实矩形分割；所有结果求和回同一 g16 key 网格。内容温度固定为 0。', '',
        '使用三个旧 calibration 场景各自清单中的第一个相机对（ordinal 0），并额外构造恒定深度、同相机的解析平面。未打开任何 labels.npz、sensor 图像或测试数据。', '',
        '| Operator | Real mean L1 | Real max L1 | Real mean JS | Max raw split defect |',
        '|---|---:|---:|---:|---:|']
    for method,s in summaries.items():
        body.append(f"| {method} | {s['real_mean_normalized_l1']:.8g} | {s['real_max_normalized_l1']:.8g} | {s['real_mean_normalized_js']:.8g} | {s['max_raw_cell_split_defect']:.8g} |")
    body+=['','独立 NumPy 实现 area-scaled Gaussian overlap；SciPy 的二维矩形积分提供独立参考；候选 B 的正值积分算子在 CPU float64 下单独对照参考。上述差异描述将矩形 key 近似为高斯后产生的分割依赖性，不能证明该缺陷是实际任务误差的主因，也不能证明 B 会通过已设定的精度、定位和效率门槛。','',
        '这个结果只能支持执行已经预声明的 B 对照实验；B 仍需与同样使用真实矩形积分的 point/quad4/quad16 对照公平比较。', '',
        '归一化比较只在两个 FOV 原始质量都大于 1e-12 的行上定义；原始质量误差保留所有 query。每行是否可归一化、原始/归一化概率、L1 和 JS 均保存为 NPZ/JSON，避免尾部精度造成虚假的巨大相对误差。', '',
        f"数据与源码 hash、精确 pair 身份在 `{args.out/'lock.json'}`；原始分数在 `summary.json` 与四个 NPZ。最大 B-vs-SciPy 绝对误差为 {max(r['max_absolute_actual_vs_scipy'] for r in all_checks):.4g}。"]
    args.doc.parent.mkdir(parents=True,exist_ok=True);args.doc.write_text('\n'.join(body)+'\n')
    write(args.out/'completion.json',dict(files={str(p.relative_to(args.out)):sha(p)
        for p in args.out.rglob('*') if p.is_file()},doc=str(args.doc),doc_sha256=sha(args.doc),
        source_hashes=source_hashes,seconds=time.time()-started,label_archives_read=False))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
