import unittest
import numpy as np
import torch
from experiments.rendered_scene import pair
from experiments.rgb_geometry_probe import RGBGeometry,observation_from_rgb


class RGBGeometryTest(unittest.TestCase):
    def test_inference_geometry_does_not_read_gt_buffers(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        view,_,gt=pair(902000)
        changed={**view,'depth':view['depth']*1.7,'normal':-view['normal'],'points':view['points']*1.9}
        model=RGBGeometry()
        cx,cy=gt['center_x'],gt['center_y']
        a=observation_from_rgb(model,view,cx,cy,None,'rgb_pred')
        b=observation_from_rgb(model,changed,cx,cy,None,'rgb_pred')
        for field in ['mean','normal','cov','uncertainty','residual','sigma','homography','support_error']:
            self.assertTrue(np.array_equal(a[field],b[field]),field)


if __name__=='__main__':
    unittest.main()
