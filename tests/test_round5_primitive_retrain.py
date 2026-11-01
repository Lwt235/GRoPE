"""Focused statistical and physical-unit checks for the isolated legacy audit."""
import numpy as np
import torch

from experiments.round5_primitive_retrain import scene_macro_statistics, state_error
from models.gaussian_spectral_pe import GaussianSpectralMatcher


def test_macro_bootstrap_weights_places_equally_and_pairs_seeds():
    # Unequal rows per place must not turn the scene-macro target into row mean.
    ids = [{"scene": "a", "q_id": 1}] * 2 + [{"scene": "b", "q_id": 3}] * 6
    center = np.zeros((5, 8))
    full = np.broadcast_to([1., 1., 2., 2., 2., 2., 2., 2.], (5, 8))
    stats = scene_macro_statistics(full, center, ids, bootstrap=1000)
    assert stats["mean"] == 1.5
    assert np.mean(full-center) == 1.75
    assert stats["leave_one_place_out"] == {"a": 2., "b": 1.}
    assert stats["ci95_seed_place_camera"] == [1., 2.]
    reverse = scene_macro_statistics(center, full, ids, bootstrap=1000)
    assert reverse["mean"] == -1.5
    np.testing.assert_allclose(reverse["ci95_seed_place_camera"],
                               -np.array(stats["ci95_seed_place_camera"])[::-1])


def test_physical_center_component_is_scaled_then_common_floor_added_once():
    model = GaussianSpectralMatcher("gaussian200_full", 256, .25)
    eye = torch.eye(2)
    centers = torch.zeros(1, 3, 2)
    fullC = (eye * .3).expand(1, 3, 2, 2)
    centerC = torch.tensor([[.1, .01], [.01, .2]]).expand(1, 3, 2, 2)
    source = {"pixq": centers, "k_to_q": centers,
              "projected_qcov": fullC, "projected_kcov": fullC}
    changed = dict(source, projected_qcov=centerC, projected_kcov=centerC)
    mq, cq, mk, ck = model.positional_inputs(changed)
    torch.testing.assert_close(cq, centerC*2 + eye*.25**2)
    torch.testing.assert_close(ck, centerC*2 + eye*.25**2)
    assert torch.equal(mq, source["pixq"]) and torch.equal(mk, source["k_to_q"])


def test_replication_requires_exact_tensor_equality():
    a = {"weights": torch.tensor([1., 2.])}
    b = {"weights": a["weights"].clone()}
    assert state_error(a, b) == {}
    b["weights"][0] = torch.nextafter(b["weights"][0], torch.tensor(float("inf")))
    assert "weights" in state_error(a, b)
