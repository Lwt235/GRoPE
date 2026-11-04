"""Fixed analytic counterexample for the scope of the existing 11D relation.

No data, predictions, training or model-source changes. This is not a task gate.
"""
import argparse
from fractions import Fraction as F
from itertools import product
import time
import numpy as np
import torch
from experiments.round65_common import ROOT, OUT, checked, read, write, sha
from models.round58_support_pe import SupportPosition
from models.round65_actual_gs import DirectPosition, common_features

DEST = OUT/'relation_information_diagnostic1'
PLAN = DEST/'plan.json'


def plan():
    checked()
    if PLAN.exists():
        raise ValueError('preserve the existing prospective fixture')
    write(PLAN, dict(created_unix=time.time(), source_sha256=sha(__file__),
        original_lock_sha256=sha(OUT/'lock.json'),
        hypothesis='The 11D pair relation can map distinct Gaussian pair configurations to identical descriptors, even after quotienting common orthogonal transforms.',
        competing_explanation='The proposed collision is only a shared rigid/reflection transform, or the actual 11D implementation distinguishes the pairs.',
        fixture=dict(A_diagonal=[1, 2, 3], B_precision=[[2, 0, 1], [0, 3, 1], [1, 1, 4]],
            B_numerator=[[11, 1, -3], [1, 7, -2], [-3, -2, 6]], B_denominator=19,
            offsets=[[1, 1, 0], [1, -1, 0]], center_scale='1/5', covariance_scale='1/25'),
        fixed_comparison='Exact rational pair terms and all common orthogonal stabilizers, then the original MLP inputs, direct descriptors and common content on these two fixed pairs.',
        decision_if_verified='Record a pair-descriptor information limitation. Do not infer task harm, whole-model indistinguishability or that an extra invariant improves accuracy. Keep the full study unchanged.',
        decision_if_not_verified='Reject/correct this proposed counterexample before making a representation claim; do not change the full study.',
        scope='Representation diagnostic only, not a new model, task benchmark, acceptance gate or fresh confirmation. All task comparisons remain pending.',
        task_data_access=False, original_protocol_changed=False, goal_complete=False))
    print('RELATION_INFORMATION_PLAN', sha(PLAN), flush=True)


def run():
    checked(); p = read(PLAN)
    if p['source_sha256'] != sha(__file__) or p['original_lock_sha256'] != sha(OUT/'lock.json'):
        raise ValueError('diagnostic or original identity changed')
    if (DEST/'result.json').exists():
        raise ValueError('preserve the completed diagnostic')
    f = p['fixture']; diag = f['A_diagonal']; pb = f['B_precision']
    b = [[F(v, f['B_denominator']) for v in row] for row in f['B_numerator']]
    offsets = f['offsets']
    # Sylvester's leading minors for the fixed precision are 2, 6, 19.
    assert pb == [[2, 0, 1], [0, 3, 1], [1, 1, 4]]
    assert all(sum(F(pb[i][k])*b[k][j] for k in range(3)) == int(i == j)
               for i in range(3) for j in range(3))
    def quad(d, m):
        return sum(F(d[i])*m[i][j]*d[j] for i in range(3) for j in range(3))
    ai = [[F(int(i == j), diag[i]) for j in range(3)] for i in range(3)]
    terms = [dict(distance_squared=sum(F(x*x) for x in d), mi=quad(d, ai), mj=quad(d, pb),
        trace_i_j=sum(b[k][k]/diag[k] for k in range(3)),
        trace_j_i=sum(F(pb[k][k]*diag[k]) for k in range(3))) for d in offsets]
    assert terms[0] == terms[1]  # Both spectra also agree because A and B are unchanged.
    # Distinct A eigenvalues force a stabilizing orthogonal map to be diagonal +/-1.
    stabilizers = [signs for signs in product((-1, 1), repeat=3)
                  if all(signs[i]*pb[i][j]*signs[j] == pb[i][j]
                         for i in range(3) for j in range(3))]
    congruent = any([s[k]*offsets[0][k] for k in range(3)] == offsets[1] for s in stabilizers)
    assert stabilizers == [(-1, -1, -1), (1, 1, 1)] and not congruent
    # Physical directional variance distinguishes the configurations exactly.
    unit_variances = [F(f['covariance_scale'])*quad(d, b)/sum(F(x*x) for x in d)
                      for d in offsets]
    assert unit_variances[0] != unit_variances[1]
    torch.set_num_threads(1); torch.manual_seed(65011)
    centers = torch.zeros(2, 2, 3, dtype=torch.float64)
    centers[:, 1] = float(F(f['center_scale']))*torch.tensor(offsets, dtype=torch.float64)
    covariance = float(F(f['covariance_scale']))*torch.stack((
        torch.diag(torch.tensor(diag, dtype=torch.float64)),
        torch.tensor([[float(v) for v in row] for row in b], dtype=torch.float64)))
    covariance = covariance.unsqueeze(0).expand(2, -1, -1, -1)
    captured = []
    position = SupportPosition('invariant_mlp').double()
    hook = position.mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    with torch.no_grad():
        relative = position(centers, covariance)
        direct = DirectPosition('direct_mlp').double().descriptors(centers, covariance)
        content = common_features(centers, covariance, torch.zeros(2, 2, 4, dtype=torch.float64))
    hook.remove()
    features = captured[0]
    np.testing.assert_allclose(features[0].numpy(), features[1].numpy(), rtol=1e-13, atol=1e-13)
    assert not torch.equal(direct[0], direct[1]) and not torch.equal(content[0], content[1])
    result = dict(plan_sha256=sha(PLAN), observed_unix=time.time(),
        exact_dimensionless_terms=[{k: str(v) for k, v in row.items()} for row in terms],
        common_orthogonal_stabilizers=stabilizers, orthogonally_congruent=congruent,
        directional_variance_exact=[str(v) for v in unit_variances],
        original_11d_inputs=features[:, 0, 1].tolist(),
        original_11d_max_abs_difference=float((features[0]-features[1]).abs().max()),
        untrained_relative_output_max_abs_difference=float((relative[0]-relative[1]).abs().max()),
        direct_descriptor_max_abs_difference=float((direct[0]-direct[1]).abs().max()),
        common_content_max_abs_difference=float((content[0]-content[1]).abs().max()),
        interpretation='The pair PE is not information-complete modulo common orthogonal transforms. The full classifier still receives distinguishing content; this does not establish whole-model collisions, task harm, or a benefit from adding features.',
        original_protocol_changed=False, task_effect_measured=False, fresh_confirmation=False,
        scientific_review_pending=True, goal_complete=False)
    write(DEST/'result.json', result)
    print('RELATION_INFORMATION_COMPLETE', sha(DEST/'result.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('plan', 'run'))
    args = parser.parse_args(); (plan if args.action == 'plan' else run)()
