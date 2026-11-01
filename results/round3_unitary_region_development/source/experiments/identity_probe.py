"""Paired random identity control with train-only marginal/noise calibration.

Uses the old matcher unchanged except for adding a distance bias. Candidate
permutation uses generator correspondence, never enters the matcher as a feature.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments import matching_probe as training
from experiments import synthetic
from models.geometry import logcov_descriptor, pairwise_squared


METHODS = ['center','rope3d_axis','rope3d_axis_normal','rope3d_axis_logcov',
           'rope3d_axis_paired_random','rope3d_axis_independent_random']
SCENARIOS = {'id': {}, 'repeated_geometry': {'region_shared':True},
             'planar_geometry': {'shared_normal':True,'shared_scale':True}}
_base_dataset = synthetic.make_dataset
_base_matcher = training.Matcher
_calibration = None


def calibrate_random():
    data = _base_dataset(51771,512)
    a,b = logcov_descriptor(data['cq']), logcov_descriptor(data['ck'])
    b = b.gather(1,data['labels'][...,None].expand_as(b))
    residual = ((a-b).square().mean((0,1))/2).clamp_min(1e-6)
    marginal = torch.cat([a,b]).var((0,1),unbiased=False)
    latent = (marginal-residual).clamp_min(1e-6)
    return {'latent_std':latent.sqrt(), 'noise_std':residual.sqrt(),
            'marginal_std':marginal.sqrt(), 'calibration_seed':51771,
            'interpretation':'Paired 6D Gaussian identity; diagonal marginal and per-coordinate correspondence residual variance matched to log-covariance using a separate calibration split only. Higher-order distributions are not identical.'}


def dataset(seed,scenes,tokens=32,scenario='id'):
    global _calibration
    if _calibration is None:
        _calibration=calibrate_random()
    data=_base_dataset(seed,scenes,tokens,scenario)
    gen=torch.Generator().manual_seed(seed+712933)
    latent=torch.randn(scenes,tokens,6,generator=gen)*_calibration['latent_std']
    q=latent+torch.randn(scenes,tokens,6,generator=gen)*_calibration['noise_std']
    k=latent+torch.randn(scenes,tokens,6,generator=gen)*_calibration['noise_std']
    independent=torch.randn(scenes,tokens,6,generator=gen)*_calibration['marginal_std']
    permutation=data['labels'].argsort(-1)
    k=k.gather(1,permutation[...,None].expand_as(k))
    independent=independent.gather(1,permutation[...,None].expand_as(independent))
    data['paired_random']=pairwise_squared(q,k)/6
    data['independent_random']=pairwise_squared(q,independent)/6
    return data


class IdentityMatcher(_base_matcher):
    def forward(self,data):
        score=super().forward(data)
        field={'rope3d_axis_paired_random':'paired_random',
               'rope3d_axis_independent_random':'independent_random'}.get(self.method)
        if field:
            score=score-F.softplus(self.shape_weight)*data[field]
        return score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='results/round2_identity')
    parser.add_argument('--seeds',default='11,22,33,44,55')
    parser.add_argument('--steps',type=int,default=400)
    args=parser.parse_args()
    torch.set_num_threads(1)
    synthetic.SCENARIOS.update(SCENARIOS)
    training.SCENARIOS=SCENARIOS
    training.make_dataset=dataset
    training.Matcher=IdentityMatcher
    args.methods=','.join(METHODS)
    args.batch,args.val_every,args.train_scenes,args.val_scenes,args.test_scenes=16,50,512,96,192
    training.run(args)
    calibration={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in _calibration.items()}
    Path(args.out,'identity_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')


if __name__=='__main__':
    main()
