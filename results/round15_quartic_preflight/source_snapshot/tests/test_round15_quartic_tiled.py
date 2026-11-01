import numpy as np
import pytest
import torch
from models.round15_quartic_tiled import integration_tiles,quartic_cell_mass
from models.round15_quartic_pe import quartic_cell_mass as reference48


torch.set_num_threads(1)


def test_native_grids_have_same432_integration_rectangles():
    expected=np.array([[x-.5,y-.5,x+7.5,y+7.5] for y in range(0,144,8)
                       for x in range(0,192,8)])/8
    expected=expected[np.lexsort((expected[:,0],expected[:,1]))]
    for stride in (8,16,24):
        bounds=torch.tensor([[x-.5,y-.5,x+stride-.5,y+stride-.5] for y in range(0,144,stride)
                             for x in range(0,192,stride)])/8
        tiles,owner=integration_tiles(bounds)
        assert len(tiles)==432 and len(torch.unique(owner))==144*192//stride**2
        result=tiles.numpy();result=result[np.lexsort((result[:,0],result[:,1]))]
        np.testing.assert_allclose(result,expected,atol=0,rtol=0)


def test_original_key_shape_probability_and_mixed_rectangles():
    mean=torch.tensor([[.4375,.4375],[-.0625,.4375],[-.0625,-.0625]],dtype=torch.float64)
    cov=torch.zeros(3,2,2,dtype=torch.float64)
    boxes=torch.tensor([[-.0625,-.0625,.9375,.9375],[-.0625,-.0625,1.9375,1.9375],
                        [-.0625,-.0625,2.9375,2.9375],[-.1,.2,1.7,2.8]],dtype=torch.float64)
    p=quartic_cell_mass(mean,cov,boxes,bandwidth=.25)
    r=reference48(mean,cov,boxes,bandwidth=.25)
    assert p.shape==(3,4)
    torch.testing.assert_close(p,r,atol=1e-8,rtol=1e-8)
    tiles,owner=integration_tiles(boxes)
    areas=(tiles[:,2:]-tiles[:,:2]).prod(-1)
    sums=torch.zeros(4,dtype=areas.dtype).index_add(0,owner,areas)
    torch.testing.assert_close(sums,(boxes[:,2:]-boxes[:,:2]).prod(-1),atol=1e-14,rtol=0)


@pytest.mark.parametrize('mu',[[.2,.6],[0.,.6],[0.,0.]])
def test_tiled_gradient_including_original_bounds(mu):
    m=torch.tensor([mu],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([[[.5,.1],[-.2,.6]]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[0.,0.,2.7,2.6]],dtype=torch.float64,requires_grad=True)
    h=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda m,a,b,h:quartic_cell_mass(m,a@a.transpose(-1,-2),b,bandwidth=h),
                                   (m,a,b,h),atol=3e-6,rtol=2e-4)


def test_no_floor_native_shape_and_invalid_extent():
    m=torch.tensor([[-128.,8.]]);c=torch.zeros(1,2,2);b=torch.tensor([[0.,0.,3.,3.]])
    p=quartic_cell_mass(m,c,b,bandwidth=.25)
    assert p.shape==(1,1) and p.item()==0
    with pytest.raises(ValueError):integration_tiles(torch.tensor([[0.,0.,3.1,1.]]))
