"""Summarize completed paired GPU measurements without claiming exclusive timings."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import time

from experiments.round65_common import ROOT, read, write, sha
from experiments.efficiency_sidecar.check_cpu import VARIANTS, ROLES


def run(out):
    r = read(out / 'result.json')
    assert r['state'] == 'complete'
    assert sha(ROOT / 'experiments/efficiency_sidecar/benchmark_gpu.py') == r['benchmark_source_sha256']
    groups = defaultdict(list)
    for row in r['measurements']:
        groups[row['role'], row['batch'], row['tokens'], row['mode'], row['variant']].append(row)
    summary = []
    for (role, batch, tokens, mode, variant), rows in groups.items():
        assert len(rows) == 4 and sorted(x['block'] for x in rows) == [0, 1, 2, 3]
        values = [x['cuda_ms'] for x in rows]
        summary.append(dict(role=role, batch=batch, tokens=tokens, mode=mode, variant=variant,
            median_cuda_ms=statistics.median(values), min_cuda_ms=min(values), max_cuda_ms=max(values),
            median_wall_ms=statistics.median(x['wall_ms'] for x in rows),
            peak_allocated_bytes=max(x['peak_bytes'] for x in rows)))
    index = {(x['role'], x['batch'], x['tokens'], x['mode'], x['variant']): x for x in summary}
    comparisons = []
    for role, batch, tokens, mode, candidate in groups:
        if candidate == 'original':
            continue
        reference = 'original' if candidate == 'factored' else 'factored'
        left = index[role, batch, tokens, mode, reference]
        right = index[role, batch, tokens, mode, candidate]
        a = {x['block']: x for x in groups[role, batch, tokens, mode, reference]}
        b = {x['block']: x for x in groups[role, batch, tokens, mode, candidate]}
        comparisons.append(dict(role=role, batch=batch, tokens=tokens, mode=mode, candidate=candidate,
            reference=reference, time_reduction_percent=100*(1-right['median_cuda_ms']/left['median_cuda_ms']),
            peak_reduction_percent=100*(1-right['peak_allocated_bytes']/left['peak_allocated_bytes']),
            paired_block_time_reduction_percent=[100*(1-b[k]['cuda_ms']/a[k]['cuda_ms']) for k in range(4)]))
    write(out / 'summary.json', dict(groups=summary, comparisons=comparisons,
        performance_scope='Shared GPU engineering pilot; not an exclusive or end-to-end throughput result.',
        result_sha256=sha(out / 'result.json')))
    tables = []
    for batch, tokens in ((4, 512), (1, 1024)):
        lines = []
        for role in ROLES:
            for mode in ('inference', 'forward_backward', 'optimizer_step'):
                cells = [index[role, batch, tokens, mode, v] for v in VARIANTS]
                numbers = ' | '.join(f"{v['median_cuda_ms']:.2f} [{v['min_cuda_ms']:.2f},{v['max_cuda_ms']:.2f}]" for v in cells)
                lines.append(f'| {role} | {mode} | {numbers} |')
        tables.append(f'''### B={batch}, N={tokens}

| 方法 | 模式 | 原版 | 已有投影重排 | 加新几何路径 | 再加激活重算 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}
''')
    rows = []
    for role in ROLES:
        values = [index[role, 4, 512, 'optimizer_step', v] for v in VARIANTS]
        rows.append(f"| {role} | " + ' | '.join(f"{v['peak_allocated_bytes']/2**30:.3f}" for v in values) + ' |')
    f = {v: index['relative5', 4, 512, 'optimizer_step', v] for v in VARIANTS}
    geom = 100*(1-f['geometry_factored']['median_cuda_ms']/f['factored']['median_cuda_ms'])
    rec = 100*(1-f['geometry_factored_recompute']['median_cuda_ms']/f['factored']['median_cuda_ms'])
    memory = (f['factored']['peak_allocated_bytes']-f['geometry_factored_recompute']['peak_allocated_bytes'])/2**30
    main_lock = read(ROOT / 'results/native_source_diversity/attempt1/lock.json')
    assert all(sha(p) == d for p, d in main_lock['source_sha256'].items() if p.endswith('.py'))
    current = read(ROOT / 'results/native_source_diversity/attempt1/runs/distance_mlp/22/progress.json')
    first = read(out / 'main_progress_first_observation.json')['progress']
    write(out / 'main_progress_after.json', dict(observed_unix=time.time(), progress=current,
        steps_since_first_observation=current['step']-first['step'],
        main_frozen_python_sources_unchanged=True,
        slowdown_attribution='No matched control interval; continued progress does not prove zero slowdown.'))
    report = f'''# 用户授权的共享 GPU 效率初测

**完成全部比较，未改变主实验代码、配置或 checkpoint。** 在 {r['gpu_name']}（GPU 4）上与
已有 Distance seed22 训练共享，单 CPU 线程、nice=15，PyTorch 分配上限 6 GiB。
共 {len(r['measurements'])} 个计时组、{len(r['checks'])} 个输出/梯度核对，耗时 {r['elapsed_seconds']:.1f} 秒。
组间休息 0.15 秒，未暂停或修改其他进程。

## 初步判断

在 B4/N512、含裁剪和 AdamW 的五维步骤中，相对已有 factored 实现，
新几何路径的中位时间减少 {geom:.2f}%，加入激活重算后减少 {rec:.2f}%（负数表示更慢）；
重算组合的自身峰值分配减少 {memory:.3f} GiB。
这些是共享状态下的工程初筛，必须结合下表块间波动判断，不据小幅中位差宣布可靠加速。
先前已有的投影重排不是本次新发现；Distance 与球形同样接受优化。

## 正确性和资源

相同旧 seed11 权重、相同旧验证输入、FP32 网络/FP64 几何、TF32 关闭。
最大 logit 绝对差 {max(x['logit_max_abs'] for x in r['checks']):.3g}，
最大全参数梯度相对 L2 差 {max(x['gradient_relative_l2'] for x in r['checks']):.3g}。
不声明长期优化轨迹一致，不把小批次数值检查当作独立任务效果确认。
含 AdamW 的测量只更新私有内存副本，没有保存科学模型。

每组记录同卡进程/显存/利用率。自身最大 PyTorch allocated 为
{max(x['peak_bytes'] for x in r['measurements'])/2**30:.3f} GiB；不包含其他进程，也不等于 nvidia-smi 的整进程占用。
主训练从首次观察到结束观察推进 {current['step']-first['step']} 步；
这证明仍在推进，**不能证明共享测试没有减速主训练**。

## 延迟

单位 ms，格式为中位数 [最小块,最大块]。四块循环平衡四种实现的次序；每组两次预热、三次测量调用。
CUDA event 时间可能包含共享调度等待，原始记录还包含 wall time。
`forward_backward` 含损失；`optimizer_step` 再含梯度裁剪和 AdamW，均不含数据加载、增强和搬运。
不同角色、不同输入规模分段测试，外部负载可能不同，不以这些跨段绝对时间断言方法间成本排名。

{chr(10).join(tables)}
## B4/N512 含优化器步骤的自身峰值分配

单位 GiB，四块取最大值。

| 方法 | 原版 | 已有投影重排 | 加新几何路径 | 再加激活重算 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 下一步

保留有价值的实现候选，暂不接入已冻结主训练。共享数据可用来发现明显退步和验证显存变化；
论文正式延迟与吞吐仍需在稳定资源下复测，并另测数据管线和完整训练吞吐。
几何重排、激活重算均是标准实现技巧，不单独宣称新位置编码或新的科学效果。

[原始记录](result.json) · [汇总与逐块配对变化](summary.json) · [主训练状态核对](main_progress_after.json)
'''
    (out / 'report.md').write_text(report)
    print(out / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
