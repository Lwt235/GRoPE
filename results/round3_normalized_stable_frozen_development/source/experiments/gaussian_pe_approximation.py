"""CPU-only Gaussian positional feature audit on development geometry.

No model is trained and no correspondence labels are inspected. The audit
compares finite spectral feature dot products with continuous BC, independently
of downstream matching. Namespace 1191 is reserved for this development audit.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.rendered_data import make_dataset
from models.gaussian_pe import spectral_rule, density_features, bhattacharyya


COUNTS = [64, 128, 256, 512, 1024, 4096]
CONDITIONS = ['oracle', 'mixed', 'normal_noise', 'scale_noise', 'wide', 'depth_fit']
OUT = Path('results/round3_approximation_fixed')
FLOORS = [.01, .0225, .0625]


def statistics(exact, approximate):
    error = (exact - approximate).abs().reshape(-1)
    top = exact.argmax(-1)
    agreement = (top == approximate.argmax(-1)).double()
    values = exact.topk(2, dim=-1).values
    margin = values[..., 0] - values[..., 1]
    unambiguous = margin > .01
    local = exact > .05
    return {
        'pairs': int(error.numel()),
        'mae': float(error.mean()),
        'p95_absolute_error': float(torch.quantile(error, .95)),
        'max_absolute_error': float(error.max()),
        'geometric_top1_agreement': float(agreement.mean()),
        'unambiguous_rows': int(unambiguous.sum()),
        'top1_agreement_margin_gt_001': float(agreement[unambiguous].mean()) if unambiguous.any() else None,
        'near_overlap_pairs': int(local.sum()),
        'near_overlap_mae_exact_gt_005': float((exact - approximate).abs()[local].mean()) if local.any() else None,
        'negative_kernel_fraction': float((approximate < 0).double().mean()),
        'approximate_min': float(approximate.min()),
        'approximate_max': float(approximate.max()),
    }


def audit_map(mq, cq, mk, ck):
    exact = bhattacharyya(mq, cq, mk, ck)
    assert torch.isfinite(exact).all()
    records = {}
    norm_errors = {}
    for count in COUNTS:
        omega, logp = spectral_rule(count, dtype=torch.float64)
        q = density_features(mq, cq, omega, logp)
        k = density_features(mk, ck, omega, logp)
        approximate = q @ k.transpose(-1, -2)
        assert torch.isfinite(approximate).all()
        records[str(count)] = statistics(exact, approximate)
        norm_errors[str(count)] = float(max((q.square().sum(-1) - 1).abs().max(), (k.square().sum(-1) - 1).abs().max()))
    return records, norm_errors


def eigen_statistics(covariance):
    eigenvalues = torch.linalg.eigvalsh(covariance).reshape(-1)
    return {
        'minimum': float(eigenvalues.min()),
        'p01': float(torch.quantile(eigenvalues, .01)),
        'median': float(eigenvalues.median()),
        'p99': float(torch.quantile(eigenvalues, .99)),
        'maximum': float(eigenvalues.max()),
    }


def selection(development):
    passing = []
    for count in COUNTS:
        records = [value['records'][str(count)] for value in development.values()]
        if all(v['mae'] <= .01 and v['p95_absolute_error'] <= .03 and v['max_absolute_error'] <= .1 and v['geometric_top1_agreement'] >= .98 for v in records):
            passing.append(count)
    return {'passing_counts': passing, 'selected_count': min(passing) if passing else None}


def mixture_diagnostic():
    """Separate defective general quadrature from the even-kernel special case."""
    def legacy_rule(count):
        u = torch.quasirandom.SobolEngine(2, scramble=True, seed=20261029).draw(count).double()
        z = math.sqrt(2) * torch.erfinv((2*u - 1).clamp(-1+1e-7, 1-1e-7))
        scales = torch.tensor([1., 4.], dtype=torch.float64)
        omega = z * scales[torch.arange(count) % 2, None]
        logp = -.5*(omega[:, None, :]/scales[None, :, None]).square().sum(-1)
        logp = logp - 2*scales.log()[None, :] - math.log(2*math.pi)
        logp = torch.logsumexp(logp, -1) - math.log(2)
        return omega, logp

    output = {'explanation': (
        'Legacy index parity confines one mixture component to positive y and the other to negative y. '
        'It is not a valid quadrature for arbitrary integrands. However, the BC kernel and feature '
        'normalizers only use integrands even under omega -> -omega; opposite half-space errors '
        'can cancel for this special case. Therefore failure of an odd test integral does not '
        'establish asymptotic bias in the BC kernel. The replacement gives each component a '
        'complete Sobol net and removes this unnecessary dependency.'), 'records': {}}
    matrix = torch.tensor([[.4, .11], [.11, .6]], dtype=torch.float64)
    delta = torch.tensor([1., -.7], dtype=torch.float64)
    exact_even = math.pi / torch.linalg.det(matrix).sqrt() * torch.exp(-.25*(delta @ torch.linalg.solve(matrix, delta)))
    for count in [256, 4096, 65536]:
        output['records'][str(count)] = {}
        for name, rule in [('legacy_parity', legacy_rule), ('complete_component_nets', lambda n: spectral_rule(n, dtype=torch.float64))]:
            omega, logp = rule(count)
            weighted = torch.exp(-torch.einsum('ni,ij,nj->n', omega, matrix, omega)-logp)
            odd = (omega[:, 1]*torch.exp(-.2*omega.square().sum(-1)-logp)).mean()
            even = (weighted*torch.cos(omega @ delta)).mean()
            if name == 'legacy_parity':
                standardized = [omega[parity::2] / scale for parity, scale in [(0, 1), (1, 4)]]
            else:
                standardized = [omega[:count//2], omega[count//2:]/4]
            output['records'][str(count)][name] = {
                'odd_integral_exact': 0., 'odd_integral_estimate': float(odd),
                'even_integral_exact': float(exact_even), 'even_integral_estimate': float(even),
                'even_integral_absolute_error': float(abs(even-exact_even)),
                'standardized_component_means': [z.mean(0).tolist() for z in standardized],
                'standardized_component_positive_y_fraction': [float((z[:, 1] > 0).double().mean()) for z in standardized],
            }
    return output


def run():
    torch.set_num_threads(1)
    start = time.time()
    source_names = ['models/gaussian_pe.py', 'experiments/rendered_data.py',
                    'experiments/rendered_scene.py', 'experiments/gaussian_pe_approximation.py']
    source_contents = {name: Path(name).read_bytes() for name in source_names}
    result = {
        'protocol': {
            'purpose': 'development-only approximation audit; no training, labels, or confirmatory test data',
            'namespace': 1191,
            'scene_indices': list(range(4)),
            'conditions': CONDITIONS,
            'counts': COUNTS,
            'dtype': 'float64',
            'device': 'cpu',
            'query_covariance': 'I / 12 in patch-grid coordinates',
            'key_covariance': 'projected_kcov',
            'resolution_floor_audit': FLOORS,
            'resolution_floor_semantics': 'Add floor*I to both query and key covariance. This changes the target continuous kernel and is not a purely numerical trick.',
            'selection_rule': {
                'maximum_condition_mae': .01,
                'maximum_condition_p95_absolute_error': .03,
                'maximum_condition_max_absolute_error': .1,
                'minimum_condition_geometric_top1_agreement': .98,
                'rule': 'smallest listed M passing every condition; no claim if none passes',
            },
        },
        'source_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in source_contents.items()},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
        'development': {},
        'resolution_floor_audit': {str(floor): {'development': {}} for floor in FLOORS},
        'stress': {},
        'mixture_diagnostic': mixture_diagnostic(),
    }
    cache = {}
    for condition in CONDITIONS:
        data, _ = make_dataset(1191, 4, condition, cache)
        mq, mk = data['pixq'].double(), data['k_to_q'].double()
        cq = torch.eye(2, dtype=torch.float64).expand(*mq.shape[:-1], 2, 2) / 12
        ck = data['projected_kcov'].double()
        records, norm_errors = audit_map(mq, cq, mk, ck)
        result['development'][condition] = {
            'records': records,
            'maximum_norm_error': norm_errors,
            'query_covariance_eigenvalues': eigen_statistics(cq),
            'key_covariance_eigenvalues': eigen_statistics(ck),
            'projected_key_coordinate_min': mk.amin(dim=(0, 1)).tolist(),
            'projected_key_coordinate_max': mk.amax(dim=(0, 1)).tolist(),
        }
        print(condition, {m: (round(v['mae'], 5), round(v['geometric_top1_agreement'], 4)) for m, v in records.items()}, flush=True)
        for floor in FLOORS:
            identity = torch.eye(2, dtype=torch.float64) * floor
            floor_records, floor_norms = audit_map(mq, cq + identity, mk, ck + identity)
            result['resolution_floor_audit'][str(floor)]['development'][condition] = {
                'records': floor_records,
                'maximum_norm_error': floor_norms,
                'query_covariance_eigenvalues': eigen_statistics(cq + identity),
                'key_covariance_eigenvalues': eigen_statistics(ck + identity),
            }

    # Fixed stress locations and covariances are independent of every benchmark
    # scene. They diagnose the valid scale range rather than tune to test data.
    generator = torch.Generator().manual_seed(202610291191)
    offsets = torch.rand(192, 2, generator=generator, dtype=torch.float64) * 16 - 8
    offsets = torch.cat([torch.zeros(1, 2, dtype=torch.float64), offsets])
    mq = torch.zeros(1, 1, 2, dtype=torch.float64)
    mk = offsets[None]
    angle = torch.tensor(np.pi / 4, dtype=torch.float64)
    rotation = torch.stack([torch.stack([angle.cos(), -angle.sin()]), torch.stack([angle.sin(), angle.cos()])])
    cases = {f'isotropic_variance_{v:g}': torch.eye(2, dtype=torch.float64) * v for v in [1e-6, 1e-4, 1e-3, 1e-2, 1/12, 1., 10., 100.]}
    for low, high in [(1e-6, 1/12), (1e-3, 1.), (1e-3, 100.)]:
        cases[f'anisotropic_eigenvalues_{low:g}_{high:g}'] = rotation @ torch.diag(torch.tensor([low, high], dtype=torch.float64)) @ rotation.T
    for name, covariance in cases.items():
        for query_kind in ['same_support', 'fixed_query_support']:
            query_covariance = covariance if query_kind == 'same_support' else torch.eye(2, dtype=torch.float64) / 12
            cq = query_covariance.expand(1, 1, 2, 2)
            ck = covariance.expand(1, len(offsets), 2, 2)
            records, norm_errors = audit_map(mq, cq, mk, ck)
            result['stress'][f'{name}/{query_kind}'] = {
                'records': records,
                'maximum_norm_error': norm_errors,
                'query_covariance_eigenvalues': eigen_statistics(cq),
                'key_covariance_eigenvalues': eigen_statistics(ck),
            }

    result['development_selection'] = selection(result['development'])
    for value in result['resolution_floor_audit'].values():
        value['selection'] = selection(value['development'])
    result['elapsed_seconds'] = time.time() - start
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in source_contents.items():
        target = OUT / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2))
    lines = [
        '# Gaussian PE Fourier approximation audit', '',
        'Development namespace 1191 only, four scenes per condition. No training or correspondence labels are used. Query covariance is I/12; key covariance is projected_kcov. All arithmetic is float64 on CPU.', '',
        '| Condition | M | MAE | 95th percentile absolute error | Maximum absolute error | Geometric top-1 agreement | Negative kernel fraction |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for condition, value in result['development'].items():
        for count, r in value['records'].items():
            lines.append(f"| {condition} | {count} | {r['mae']:.5f} | {r['p95_absolute_error']:.5f} | {r['max_absolute_error']:.5f} | {100*r['geometric_top1_agreement']:.2f}% | {100*r['negative_kernel_fraction']:.2f}% |")
    selected = result['development_selection']['selected_count']
    lines += ['', f'Selection: {selected if selected is not None else "none of the audited M values passes"}. The numerical rule requires, in every development condition, MAE ≤ 0.01, 95th percentile ≤ 0.03, maximum error ≤ 0.1, and geometric top-1 agreement ≥ 98%. This checks approximation quality, not task performance.', '',
              'Geometric top-1 compares the maxima of the exact and approximate kernels; it does not read correspondence ground truth. The JSON also records margins, overlapping-pair error, covariance eigenvalue ranges, exact unit-norm checks, source hashes, and stress tests.', '',
              'Stress support variances range from 1e-6 to 100 and include rotated anisotropic ellipses. These artificial cases diagnose where the finite proposal fails; they are not benchmark test scenes and are not used to choose task hyperparameters. Unit normalization alone does not guarantee an accurate BC kernel.', '',
              '| Stress support | M | MAE | Maximum absolute error |', '|---|---:|---:|---:|']
    for name, value in result['stress'].items():
        for count in ['256', '4096']:
            r = value['records'][count]
            lines.append(f"| {name} | {count} | {r['mae']:.5f} | {r['max_absolute_error']:.5f} |")
    lines += ['',
              'Interpretation: M=256 preserves most geometric top-1 choices but leaves substantial signed oscillation on distant pairs. Near-overlap error is much smaller than global error. Increasing M to 4096 resolves most common supports, while extremely thin projected supports in wide/depth_fit still fail the maximum-error criterion. The JSON records their covariance ranges.', '',
              'For thin supports, true Fourier energy extends past the fixed proposal\'s well-sampled range. Finite-map normalization can then inflate the sampled low-frequency energy and overestimate overlap with wider supports. Stress cases show that unit norm alone is not a sufficient approximation guarantee. Investigate exact BC first, then improve quadrature or state a physically justified resolution floor and audit it again.', '',
              'This is an approximation diagnostic. Poor kernel error cannot be interpreted as a failure of the continuous BC hypothesis; a downstream gain from a poor approximation must be described as a different finite positional kernel, not automatically attributed to true Gaussian overlap.', '']
    lines += ['## Complete component nets and legacy parity', '',
              result['mixture_diagnostic']['explanation'], '',
              '| M | Rule | Odd integral, exact 0 | Even integral absolute error |',
              '|---:|---|---:|---:|']
    for count, rules in result['mixture_diagnostic']['records'].items():
        for name, value in rules.items():
            lines.append(f"| {count} | {name} | {value['odd_integral_estimate']:.6f} | {value['even_integral_absolute_error']:.6f} |")
    lines += ['', '## Resolution floor changes the target kernel', '',
              'Both query and key covariance receive floor*I before computing either exact BC or its feature approximation. This is a deliberately different, blurred support kernel, not a numerical equivalence. Selection still uses only development geometry.', '',
              '| Added variance floor | Smallest passing M |', '|---:|---:|']
    for floor, value in result['resolution_floor_audit'].items():
        lines.append(f"| {floor} | {value['selection']['selected_count']} |")
    lines += ['', 'Detailed error metrics, covariance ranges, norm checks and each passing-M list are in audit.json. Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.gaussian_pe_approximation`. The earlier legacy-rule audit remains in results/round3_approximation.', '']
    (OUT / 'audit.md').write_text('\n'.join(lines))
    print(json.dumps(result['development_selection']), flush=True)
    print(f'elapsed_seconds={result["elapsed_seconds"]:.2f}', flush=True)


if __name__ == '__main__':
    run()
