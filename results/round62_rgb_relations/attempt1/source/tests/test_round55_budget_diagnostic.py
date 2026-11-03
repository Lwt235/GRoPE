import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from experiments.round55_optimization import learning_rate,sampling,save_resume,restore_resume
from models.round55_budget_diagnostic import make_model,ROLES
from models.round52_camera_frame import CameraFrameMatcher
from models.round53_ray_frame import RayFrameMatcher
from models.round54_calibrated_plane import CalibratedPlaneMatcher
from tests.test_round53_ray_frame import fixture


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)

    def test_original_schedule_and_batch_prefix(self):
        for step in range(1000):
            old=.001*(.2+.8*(1+math.cos(math.pi*step/1000))/2)
            self.assertEqual(learning_rate(step),old)
        self.assertEqual(learning_rate(1000),.0002)
        self.assertEqual(learning_rate(4999),.0002)
        for seed in (11,22,33):
            old=np.random.default_rng(seed+50000).integers(0,576,size=(1000,8),dtype=np.int64)
            full=sampling(seed,576)
            np.testing.assert_array_equal(full[:1000],old)
            self.assertEqual(full.shape,(5000,8))

    def test_exact_original_classes_functions_and_common_state(self):
        data={k:v.float() for k,v in fixture()[0].items()};shared=None
        for role in ROLES:
            factory=RayFrameMatcher if role.startswith('ray_') else CalibratedPlaneMatcher if role in ('camera_offset','camera_plane') else CameraFrameMatcher
            torch.manual_seed(11);model=make_model(role)
            torch.manual_seed(11);old=factory(role)
            self.assertIs(type(model),type(old))
            state={k:v for k,v in model.state_dict().items() if not k.startswith('field.')}
            if shared is None:shared=state
            else:
                for key in state:torch.testing.assert_close(state[key],shared[key],atol=0,rtol=0)
            with torch.no_grad():torch.testing.assert_close(model(data),old(data),atol=0,rtol=0)

    def test_optimizer_and_random_state_resume_matches_continuation(self):
        data={k:v.float() for k,v in fixture()[0].items()};device=torch.device('cpu')
        for role in ('camera_plane','ray_gaussian'):
            torch.manual_seed(19);model=make_model(role)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            def update(model,optimizer,step):
                model.train();optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(step)
                logits=model(data)
                # A fresh stochastic probe makes loss of RNG state observable.
                loss=(logits-torch.randn_like(logits)).square().mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            update(model,optimizer,998);update(model,optimizer,999)
            with tempfile.TemporaryDirectory(prefix='grope_r55_test_') as temp:
                path=Path(temp)/'resume.pt';save_resume(path,model,optimizer,device,dict(step=1000))
                update(model,optimizer,1000);update(model,optimizer,1001)
                expected={k:v.clone() for k,v in model.state_dict().items()};random_after=torch.rand(3)
                restored=make_model(role);other=torch.optim.AdamW(restored.parameters(),lr=.7,weight_decay=.2)
                state=restore_resume(path,restored,other,device);self.assertEqual(state['step'],1000)
                update(restored,other,1000);update(restored,other,1001)
                for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
                torch.testing.assert_close(torch.rand(3),random_after,atol=0,rtol=0)
                self.assertTrue(all(float(s['step'])==4 for s in other.state.values()))


if __name__=='__main__':unittest.main()
