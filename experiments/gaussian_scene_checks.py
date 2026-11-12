"""Small renderer/field identity checks; these establish no downstream gain."""
import json
from pathlib import Path

import numpy as np
import torch

from models.gaussian_scene_memory import (
    contribution_weights, rasterize_field, lift_source_features, field_bytes)
from experiments.gaussian_scene_setup import OUT, ASSETS, write


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    field = dict(means=torch.tensor([[0., 0., 2.], [0., 0., 4.], [.4, .1, 3.]], device='cuda'),
                 rotations=torch.tensor([[1., 0., 0., 0.]] * 3, device='cuda'),
                 scales=torch.full((3, 3), .15, device='cuda'),
                 opacities=torch.tensor([.8, .8, .5], device='cuda'),
                 colors=torch.rand(3, 3, device='cuda'))
    c2w = torch.eye(4, device='cuda')[None]
    K = torch.tensor([[[16., 0., 4.], [0., 16., 4.], [0., 0., 1.]]], device='cuda')
    weights, alpha = contribution_weights(field, c2w, K, 8, 8)
    rgb, alpha2, _ = rasterize_field(field, c2w, K, 8, 8)
    color_error = float((weights @ field['colors'] - rgb).abs().max())
    alpha_error = float((weights.sum(-1, keepdim=True) - alpha).abs().max())
    assert color_error < 1e-5 and alpha_error < 1e-5
    assert torch.equal(alpha, alpha2)
    assert weights[0, 4, 4, 0] > weights[0, 4, 4, 1]
    feature = torch.randn(1, 8, 8, 1024, device='cuda')
    f, mass, valid = lift_source_features(field, feature, c2w, K)
    assert f.shape == (3, 1024) and torch.isfinite(f).all() and valid.all()
    field.update(features=f, content_mass=mass, has_content=valid, source_c2w=c2w, source_intrinsic=K)
    path = ASSETS / 'synthetic_field_check.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in field.items()}, path)
    restored = {k: v.cuda() for k, v in torch.load(path, map_location='cpu', weights_only=True).items()}
    repeated, _, _ = rasterize_field(restored, c2w, K, 8, 8)
    assert torch.equal(repeated, rgb)
    wrong = c2w.clone()
    wrong[:, 0, 3] += .5
    moved, _, _ = rasterize_field(restored, wrong, K, 8, 8)
    camera_change = float((moved - rgb).abs().max())
    assert camera_change > .01
    result = dict(complete=True, cuda_device=torch.cuda.get_device_name(),
                  basis_color_compositing_max_abs=color_error,
                  sum_basis_vs_alpha_max_abs=alpha_error,
                  front_gaussian_has_larger_center_contribution=True,
                  serialized_field_render_exact=True, camera_change_max_abs=camera_change,
                  lifted_features_finite=True, persistent_tensor_bytes=field_bytes(field),
                  scope='Synthetic renderer identity, front-to-back visibility and saved field only; not AnySplat or VGGT utility.')
    write(OUT / 'renderer_interface_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
