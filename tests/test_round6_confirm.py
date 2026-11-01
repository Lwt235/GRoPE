"""Offline contracts for confirmation identity and known-footprint verification."""
import copy

import pytest
import torch

from experiments.round6_confirm import verify_data
from experiments.round6_train import GRIDS


def fixture():
    groups, identities = {}, {}
    for grid in GRIDS:
        qs, ks = map(int, grid.replace('q', '').replace('k', '').split('_'))
        nq, nk = (144//qs)*(192//qs), (144//ks)*(192//ks)
        # Only verifier-consumed fields are needed; no model/GT inference.
        groups[grid] = {
            'q': torch.zeros(1, nq, 1), 'k': torch.zeros(1, nk, 1),
            'qcov': (torch.eye(2)*(qs**2/(12*64))).expand(1, nq, 2, 2).clone(),
            'kcov': (torch.eye(2)*(ks**2/(12*64))).expand(1, nk, 2, 2).clone(),
            'soft_labels': torch.zeros(1, nq, nk)}
        identities[grid] = [dict(scene='fresh/scan', q_index=30, k_index=45)]
    selection = {'scenes': [dict(scene='fresh/scan', split='test', pairs=[[30, 45]])]}
    return groups, identities, selection


def test_complete_four_grid_pair_identity_and_continuous_native_covariance():
    groups, identities, selection = fixture()
    observed = verify_data(groups, identities, selection, 'test')
    assert observed == {('fresh/scan', 30, 45)}
    torch.testing.assert_close(groups['q16_k8']['qcov'][0, 0],
        4*groups['q8_k8']['qcov'][0, 0], rtol=0, atol=0)


def test_pair_missing_or_replaced_in_one_grid_cannot_become_an_extra_sample():
    groups, identities, _ = fixture()
    identities['q16_k16'][0]['k_index'] = 90
    with pytest.raises(ValueError, match='unequal grid pair sets'):
        verify_data(groups, identities)


def test_duplicate_pairs_are_rejected_even_if_npz_axes_match_duplicate_count():
    groups, identities, _ = fixture()
    for grid in GRIDS:
        identities[grid] *= 2
        groups[grid] = {key: value.repeat(2, *([1]*(value.ndim-1)))
                        for key, value in groups[grid].items()}
    with pytest.raises(ValueError, match='duplicate data identity'):
        verify_data(groups, identities)


def test_npz_leading_axis_must_match_manifest_pair_count():
    groups, identities, _ = fixture()
    groups['q8_k8']['q'] = groups['q8_k8']['q'].repeat(2, 1, 1)
    with pytest.raises(ValueError, match='NPZ rows differ'):
        verify_data(groups, identities)


def test_labels_must_cover_actual_query_and_key_token_grid():
    groups, identities, _ = fixture()
    groups['q8_k16']['soft_labels'] = groups['q8_k16']['soft_labels'][..., :-1]
    with pytest.raises(ValueError, match='grid/data shape mismatch'):
        verify_data(groups, identities)


@pytest.mark.parametrize('field', ['qcov', 'kcov'])
def test_uniform_but_wrong_native_size_is_rejected(field):
    groups, identities, _ = fixture()
    groups['q16_k16'][field] *= .25  # A plausible accidental stride-8 footprint.
    with pytest.raises(AssertionError):
        verify_data(groups, identities)


def test_complete_export_must_equal_frozen_original_frame_selection():
    groups, identities, selection = fixture()
    wrong = copy.deepcopy(selection)
    wrong['scenes'][0]['pairs'] = [[30, 90]]
    with pytest.raises(ValueError, match='incomplete or extra selected pair set'):
        verify_data(groups, identities, wrong, 'test')
