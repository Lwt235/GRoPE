import copy

import pytest

from experiments import round10_cell_selection_repair as R


def summaries():
    result={}
    for h in R.B.WIDTHS:
        for t in R.B.TEMPERATURES:
            result[f'point_h{h}_t{t}']={'kl':.2,'pck8':.9,'by_scene':{'a':.3,'b':.1},'by_condition':{'x':.2}}
            result[f'center_h{h}_t{t}']=copy.deepcopy(result[f'point_h{h}_t{t}'])
    return result


def test_roundoff_guard_without_modifying_scores():
    value=summaries();value['center_h0.25_t0.0']['kl']+=2.e-16
    before=copy.deepcopy(value)
    result=R.guard(value)
    assert 0<result['max_absolute_difference']<1e-12
    assert value==before


def test_material_discrepancy_still_rejected():
    value=summaries();value['center_h0.25_t0.0']['by_scene']['b']+=2e-12
    with pytest.raises(ValueError,match='exceeds'):R.guard(value)


@pytest.mark.parametrize('other',[{'x':float('nan')},{'x':float('inf')},{'x':'1'},{'y':1.}])
def test_nonfinite_type_or_structure_change_rejected(other):
    with pytest.raises(ValueError):R.recursive_difference({'x':1.},other)


def test_exact_nonnumeric_and_sequence_contract():
    assert R.recursive_difference({'a':[True,'x',2.]},{'a':[True,'x',2.]})==0
    with pytest.raises(ValueError):R.recursive_difference({'a':[True,'x']},{'a':[1,'x']})


def test_repair_refuses_existing_selection_or_evaluation(tmp_path):
    R.assert_no_selection(tmp_path)
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError):R.assert_no_selection(tmp_path)
    (tmp_path/'selection.json').unlink()
    (tmp_path/'evaluation').mkdir()
    with pytest.raises(ValueError):R.assert_no_selection(tmp_path)
