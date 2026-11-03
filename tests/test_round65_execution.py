"""Properties of sampling, augmentation, evaluation and resource selection."""
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from experiments.round65_run import epoch_batches,stream_update,augment,evaluation_batch,metrics,learning_rate
from experiments.round65_audit import independent_metrics,bootstrap
from experiments.round65_common import CONFIG,VARIANTS
from experiments.round65_queue import occupancy


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_sampler_resume_and_complete_object_coverage(self):
        rng=np.random.default_rng(65011)
        first=list(epoch_batches(rng,7,batch_size=3,tokens=11))
        np.testing.assert_array_equal(np.sort(np.concatenate([b[0] for b in first])),np.arange(7))
        for _,subset,_,_ in first:
            for row in subset:self.assertEqual(len(np.unique(row)),11)
        saved=json.loads(json.dumps(rng.bit_generator.state))
        state=hashlib.sha256(b'sampler test').hexdigest()
        expected=[]
        for batch in epoch_batches(rng,7,batch_size=3,tokens=11):
            state=stream_update(state,*batch);expected.append(batch)
        other=np.random.default_rng();other.bit_generator.state=saved
        replay_state=hashlib.sha256(b'sampler test').hexdigest()
        for a,b in zip(expected,epoch_batches(other,7,batch_size=3,tokens=11)):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            replay_state=stream_update(replay_state,*b)
        self.assertEqual(state,replay_state)

    def test_rotation_moves_native_center_and_shape_together(self):
        rng=np.random.default_rng(2);q=rng.normal(size=(2,4))
        c=torch.tensor(rng.normal(size=(2,5,3)),dtype=torch.float32)
        cov=torch.diag_embed(torch.tensor(rng.uniform(.001,.1,size=(2,5,3)),dtype=torch.float64))
        jitter=rng.normal(size=(2,5,3)).astype(np.float32)*.01
        c2,s2=augment(c,cov,q,jitter)
        r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(c2.numpy(),np.einsum('bij,bnj->bni',r,c.numpy())+jitter,atol=3e-7)
        np.testing.assert_allclose(s2.numpy(),r[:,None]@cov.numpy()@r[:,None].swapaxes(-1,-2),atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(s2),np.linalg.eigvalsh(cov),atol=1e-14)

    def test_evaluation_conditions_preserve_source_and_token_correspondence(self):
        rng=np.random.default_rng(3)
        bundle=dict(centers=rng.normal(size=(2,2048,3)).astype(np.float32),
            covariance=np.broadcast_to(np.diag([.001,.002,.003]),(2,2048,3,3)).copy(),
            appearance=rng.normal(size=(2,2048,4)).astype(np.float32),ids=np.array(['synthetic/a','synthetic/b']))
        before={k:v.tobytes() for k,v in bundle.items()}
        base=evaluation_batch(bundle,0,2,'base','cpu')
        for variant in VARIANTS:
            a=evaluation_batch(bundle,0,2,variant,'cpu');b=evaluation_batch(bundle,0,2,variant,'cpu')
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
            count=256 if variant=='sparse256' else 1024 if variant=='dense1024' else 512
            self.assertEqual(a[0].shape,(2,count,3))
            if variant=='permuted':
                for x,y in zip(a,base):torch.testing.assert_close(x,y.flip(1),atol=0,rtol=0)
            if variant=='jitter02':torch.testing.assert_close(a[1],base[1],atol=0,rtol=0)
        self.assertEqual({k:v.tobytes() for k,v in bundle.items()},before)

    def test_metric_reference_and_fixed_schedule(self):
        logits=np.zeros((60,60));labels=np.arange(60)
        a,b=metrics(logits,labels),independent_metrics(logits,labels)
        for row in (a,b):
            self.assertAlmostEqual(row['ce'],np.log(60),places=13)
            self.assertAlmostEqual(row['oa'],1/60,places=13)
            self.assertAlmostEqual(row['mca'],1/60,places=13)
        logits[np.arange(60),labels]=10
        self.assertEqual(independent_metrics(logits,labels)['mca'],1.)
        interval=bootstrap(np.ones((3,60))*.75)
        np.testing.assert_allclose(interval,[.75,.75],atol=1e-14)
        steps=390
        self.assertAlmostEqual(learning_rate(CONFIG['warmup_epochs']*steps-1,steps),.001)
        self.assertAlmostEqual(learning_rate(CONFIG['epochs']*steps-1,steps),.00001)

    def test_gpu_with_live_job_is_never_idle(self):
        replies=[SimpleNamespace(stdout='0, GPU-zero, 0, 0\n1, GPU-one, 0, 0\n2, GPU-two, 500, 0\n'),
                 SimpleNamespace(stdout='GPU-one, 12345\n')]
        with patch('experiments.round65_queue.subprocess.run',side_effect=replies):
            rows=occupancy()
        self.assertEqual([r['index'] for r in rows if r['free']],[0])


if __name__=='__main__':unittest.main()
