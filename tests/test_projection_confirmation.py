"""Before new pixels: reproduce old preprocessing and preserve the input boundary."""
import unittest
import numpy as np
import torch
from experiments.round65_common import ROOT,read
from experiments.round70_nvs_data import OUT,NVSData
from experiments.round5_rgbd_data import load_sun3d_scene
from experiments.projection_confirmation_data import frame_arrays,ConfirmationData,selected_triples


class ConfirmationTests(unittest.TestCase):
    def test_preprocessing_reproduces_old_cached_pixels_and_geometry(self):
        manifest=read(OUT/'data/manifest.json');data=NVSData('fit');seen=set();count=0
        for row in manifest['frames']:
            if row['split']!='fit' or row['scene'] in seen:continue
            seen.add(row['scene']);frames,_=load_sun3d_scene(ROOT/'data/round5_sun3d',row['scene'])
            actual=frame_arrays(frames[row['index']]);index=row['bank_index']
            for key,value in actual.items():np.testing.assert_array_equal(value,data.arrays[key][index],err_msg=key)
            count+=1
        self.assertEqual(count,3)

    def test_batch_condition_and_information_boundary_unchanged(self):
        old=NVSData('val');new=ConfirmationData.__new__(ConfirmationData)
        new.arrays=old.arrays;new.split='confirmation'
        for role in ('rgb_target_relative5','observed_target_relative5'):
            for condition in ('base','extrapolate','target_zoom'):
                inputs,target=new.batch(np.array([0,8,31]),role,'cpu',condition)
                expected,labels=old.batch(np.array([0,8,31]),role,'cpu',condition)
                self.assertTrue(torch.equal(target,labels))
                self.assertEqual(set(inputs),set(expected))
                self.assertNotIn('target_rgb',inputs);self.assertNotIn('target_depth',inputs)
                if role.startswith('rgb_'):self.assertEqual(set(inputs),{'ref_rgb','K','c2w'})
                for key in inputs:self.assertTrue(torch.equal(inputs[key],expected[key]))

    def test_ordinals_preserve_frozen_triples(self):
        q=np.linspace(30,1000-91,48).astype(int)
        row={'pairs':[(int(i),int(i+o)) for i in q for o in (15,60)]}
        self.assertEqual(selected_triples(row),[(int(i),int(i+15),int(i+60)) for i in q])


if __name__=='__main__':unittest.main()
