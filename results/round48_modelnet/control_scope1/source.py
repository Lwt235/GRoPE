"""Data-free coefficient-domain examples for interpreting the fixed R48 control.

Equal parameter counts and an equal initial tangent do not make these bounded
parameterizations nested. This is an interpretation note, not a new experiment
gate, a diagnosis of trained models, or a change to the running comparison.
"""
from pathlib import Path
import time
import numpy as np
import torch

from experiments.round48_common import ROOT, checked, sha, write
from models.round47_conditional_gaussian import GaussianPosition

OUT = ROOT/'results/round48_modelnet/control_scope1'


def coefficients(kind, mean, diagonal):
    model = GaussianPosition(kind+'3d').double()
    shape = np.r_[diagonal, np.zeros(3)]
    raw = np.r_[4*np.arctanh(np.asarray(mean)/4), 3*np.arctanh(shape/3)]
    with torch.no_grad():
        model.predictor.bias.copy_(torch.as_tensor(np.tile(raw, 2)))
        precision, linear = model.coefficients(torch.zeros(1, 1, 96, dtype=torch.float64))
    return precision[0, 0, 0].numpy(), linear[0, 0, 0].numpy()


def main():
    lock = checked()
    if OUT.exists():
        raise ValueError('preserve existing control-scope evidence')
    OUT.mkdir(parents=True)
    write(OUT/'plan.json', dict(
        unix=time.time(), source_sha256=sha(__file__),
        frozen_model_sha256=lock['sources']['models/round47_conditional_gaussian.py'],
        purpose='Explain coefficient ranges of the fixed comparison before dev_eval.',
        fixtures=['Gaussian shape diagonal 1 and zero mean',
                  'Free quadratic shape diagonal -1 and zero mean',
                  'Gaussian shape diagonal .5 and mean (2,0,0), with matched free precision'],
        task_data_access=False, checkpoint_access=False, modifies_R48=False,
        is_R48_gate=False))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    free_diagonal_bounds = (-5/16, 7/16)
    free_linear_abs_bound = 4/16
    gp, gb = coefficients('conditional', [0., 0., 0.], [1., 1., 1.])
    fp, fb = coefficients('free_quadratic', [0., 0., 0.], [-1., -1., -1.])
    cp, cb = coefficients('conditional', [2., 0., 0.], [.5, .5, .5])
    mp, mb = coefficients('free_quadratic', [0., 0., 0.], [(np.e-1)/2]*3)
    # Independent closed forms for finite, specified raw-predictor outputs.
    np.testing.assert_allclose(gp, np.eye(3)*np.exp(2)/16, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gb, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fp, -np.eye(3)/16, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fb, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(cp, np.eye(3)*np.e/16, atol=1e-12, rtol=0)
    np.testing.assert_allclose(mp, cp, atol=1e-12, rtol=0)
    np.testing.assert_allclose(cb, [np.e/8, 0, 0], atol=1e-12, rtol=0)
    assert gp[0, 0] > free_diagonal_bounds[1]
    assert np.linalg.eigvalsh(fp).max() < 0
    assert cb[0] > free_linear_abs_bound
    checked()
    result = dict(
        completed=True,
        formula='shape=3*tanh(raw_shape/3), mean=4*tanh(raw_mean/4); '
                'Gaussian P=L*L^T/16, b=P*mean; free P=(I+U+U^T)/16, b=mean/16.',
        free_precision_diagonal_open_interval=free_diagonal_bounds,
        free_linear_component_open_interval=[-free_linear_abs_bound, free_linear_abs_bound],
        gaussian_precision_outside_free_diagonal_range=gp.tolist(),
        free_precision_outside_gaussian_SPD_domain=fp.tolist(),
        shared_feasible_precision=cp.tolist(),
        gaussian_linear_at_shared_precision_outside_free_range=cb.tolist(),
        conclusion='The two bounded coefficient families are not nested. '
                   'The fixed comparison changes attainable coefficient ranges and nonlinear '
                   'parameterization as well as SPD and center-precision coupling.',
        limitation='No trained coefficient distribution was inspected. These examples do not '
                   'show that range limits are active in R47/R48, explain any observed task '
                   'difference, or alter the predeclared empirical utility comparison.',
        own_derivation=True, task_data_access=False, checkpoint_access=False,
        frozen_sources_unchanged=True, modifies_R48=False, is_R48_gate=False)
    write(OUT/'result.json', result)
    print(result)


if __name__ == '__main__':
    main()
