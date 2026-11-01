"""Additional numerical diagnostics prompted by the first round."""
import json
from pathlib import Path
import torch
from models.geometry import logcov_descriptor


def run():
    torch.set_num_threads(1)
    result = []
    for gap in [0., 1e-12, 1e-8, 1e-4, .1]:
        c = torch.diag(torch.tensor([1., 1.+gap, 2.], dtype=torch.float64)).requires_grad_()
        logcov_descriptor(c).sum().backward()
        result.append({'eigenvalue_gap': gap, 'eigh_log_gradient_finite': bool(torch.isfinite(c.grad).all())})
    # Trace-normalized covariance plus scalar log-volume proxy: smooth at sphere.
    c = torch.eye(3, dtype=torch.float64, requires_grad=True)
    trace = c.trace()
    feature = torch.cat([(c/trace).flatten(), trace.log().reshape(1)])
    feature.sum().backward()
    output = {'eigendecomposition_gradient': result,
              'trace_normalized_covariance_isotropic_gradient_finite': bool(torch.isfinite(c.grad).all()),
              'interpretation': 'PyTorch eigh backward is singular at repeated eigenvalues. Log-SPD as a mathematical function is smooth there; this is an implementation issue. Pilot descriptors are precomputed and not differentiated. Use a stable spectral Frechet derivative or a smooth covariance descriptor before end-to-end prediction.'}
    Path('results/pilot/stability.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    run()
