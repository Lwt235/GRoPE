"""A data-free counterexample about transferring Gaussian natural gradients.

This does not modify R48 or test an alternative optimizer. It verifies that
preconditioning each predicted Gaussian mean's output gradient is generally not
the natural gradient of a shared predictor's weights. No dataset is opened.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round48_modelnet/preconditioning_scope1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if OUT.exists():
        raise ValueError('preserve existing scope note')
    OUT.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    plan = dict(
        purpose='Check a proposed optimizer transfer, not a model utility claim or new gate.',
        source_sha256=hashlib.sha256(source).hexdigest(), unix=time.time(),
        fixture=dict(jacobian=[[1., 0.], [0., 1.], [1., 1.]],
                     output_fisher_diagonal=[1., 4., 2.], output_loss_gradient=[1., -1., 2.],
                     common_kl_radius=.0001),
        interpretation='Three independent scalar Gaussians with fixed variances, whose means share two weights.',
        data_access=False, modifies_R48=False, fresh_confirmation_access=False)
    write(OUT/'plan.json', plan)
    (OUT/'source.py').write_bytes(source)
    fixture = plan['fixture']
    J = np.asarray(fixture['jacobian'])
    fisher = np.asarray(fixture['output_fisher_diagonal'])
    g = np.asarray(fixture['output_loss_gradient'])
    gradient = J.T@g
    pullback = J.T@(fisher[:, None]*J)
    natural = np.linalg.solve(pullback, gradient)
    output_hook = J.T@(g/fisher)
    projected_scale = float(natural@output_hook/(natural@natural))
    noncollinear_residual = float(np.linalg.norm(output_hook-projected_scale*natural))
    rows = []
    for name, direction in [('shared_weight_natural_gradient', natural),
                            ('output_precondition_then_backprop', output_hook)]:
        scale = np.sqrt(2*fixture['common_kl_radius']/(direction@pullback@direction))
        step = -scale*direction
        # Exact KL for the independent Gaussian means, not a small-step estimate.
        exact_kl = float(.5*np.sum(fisher*(J@step)**2))
        rows.append(dict(method=name, descent_direction=direction.tolist(),
                         equal_kl_step=step.tolist(), exact_kl=exact_kl,
                         linear_loss_decrease=float(-gradient@step)))
    # A single shared global learning-rate rescaling cannot remove this mismatch.
    assert noncollinear_residual > .1
    assert natural[1] < 0 < output_hook[1]
    assert all(abs(row['exact_kl']-fixture['common_kl_radius']) < 1e-15 for row in rows)
    assert rows[0]['linear_loss_decrease'] > rows[1]['linear_loss_decrease']
    code_base = ('https://github.com/jimtjames/3dGaussianPointEncoders/blob/'
                 '3e08aa732a81033f1c07c0b0533c334fe448e0d3/')
    result = dict(
        completed=True, pullback_fisher=pullback.tolist(), weight_gradient=gradient.tolist(),
        noncollinear_residual=noncollinear_residual, methods=rows,
        conclusion='Output covariance gradient hooks are not generally the shared-weight natural gradient.',
        formula='G_theta=sum_i J_i^T F_i J_i; natural=solve(G_theta,sum_i J_i^T g_i). '
                'An output hook instead gives sum_i J_i^T solve(F_i,g_i).',
        limitation='This does not prove that such a heuristic cannot improve optimization; '
                   'it does not diagnose R47/R48 or license changing their frozen optimizers.',
        sources=[
            dict(url='https://arxiv.org/html/2511.04797v1',
                 scope='Sections 3.3 and 3.4; point/patch encoder replacement and independent Gaussian natural gradients.',
                 factual_summary='The method replaces point embeddings in PointNet/Mamba3D and studies natural gradients '
                                 'and distillation. Its optimization findings concern those encoders, not our conditional attention PE.'),
            dict(url=code_base+'Mamba3D/models/Mamba3D.py#L590',
                 git_blob_sha='0f76a4ddc8cf494a9b2d816d10e06f229060fe6a',
                 factual_summary='GaussianMamba3D constructs a Gaussian patch encoder and a separate 3-to-128-to-trans_dim '
                                 'MLP positional encoder; the forward path uses both.'),
            dict(url=code_base+'PointNet/encoder.py#L1058',
                 git_blob_sha='026c30493ac0996dcb1abc23f31f69afb5666c1e',
                 factual_summary='The Mahalanobis hook multiplies gradients of independent stored Gaussian means '
                                 'by their covariance matrices.')],
        own_derivation=True, external_method_reproduced=False, official_test_access=False,
        modifies_R48=False, is_R48_gate=False, goal_complete=False)
    write(OUT/'result.json', result)
    print(json.dumps({k: result[k] for k in ('completed', 'pullback_fisher', 'weight_gradient',
                                           'noncollinear_residual', 'methods', 'conclusion')}))


if __name__ == '__main__':
    main()
