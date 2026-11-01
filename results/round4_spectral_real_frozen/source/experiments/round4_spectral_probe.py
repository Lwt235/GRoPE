"""Reuse the audited trainers with an explicit new factory and data namespace.

Old trainers/model files are left unchanged. Every invocation freezes its exact
configuration and source snapshot before training; existing output is refused.
Real evaluation is deliberately separate, after all training finishes.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.rendered_probe import source_snapshot
from models.gaussian_spectral_pe import GaussianSpectralMatcher
from models.gaussian_pe import GaussianPEMatcher


REAL_METHODS = ['gaussian_full', 'gaussian_qonly', 'gaussian_konly', 'gaussian_none',
                'gaussian_trace', 'gaussian_area', 'gaussian25_full', 'gaussian50_full',
                'gaussian200_full', 'resolvent_full', 'resolvent_none',
                'resolvent_trace', 'resolvent_area']
POINT_METHODS = ['projected_rope', 'gaussian_full', 'gaussian_none',
                 'gaussian25_full', 'gaussian50_full', 'gaussian200_full', 'resolvent_full']


def factory(method, count=256, resolution=.25):
    if method == 'projected_rope':
        return GaussianPEMatcher(method, count, resolution)
    return GaussianSpectralMatcher(method, count, resolution)


def main(args):
    out = Path(args.out)
    if (out/'manifest.json').exists() or (out/'frozen_protocol.json').exists():
        raise FileExistsError('Use a fresh output; never overwrite a recorded experiment')
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(',') if args.methods else (REAL_METHODS if args.domain == 'real' else POINT_METHODS)
    config = dict(device='cpu', phase='development' if args.domain == 'real' else 'confirmation',
        out=args.out, seeds=args.seeds, methods=','.join(methods), features=256, resolution=.25,
        steps=500 if args.domain == 'real' else 600, batch=16,
        train_scenes=192, val_scenes=48, test_scenes=96,
        val_every=20 if args.domain == 'real' else 100)
    if args.domain == 'real':
        config.update(domain='real', query_geometry='fitted', scene='teatime',
            real_root='results/round3_real_support_data_adjacent' if args.kind == 'raw'
                      else 'results/round3_frozen_visual_data')
    else:
        config.update(regimes='oracle_train,robust_train')
    protocol = dict(config=config, kind=args.kind, domain=args.domain,
        methods=methods, source_hashes=source_snapshot(out),
        question='Symmetric q/k shape controls; exponential versus exponential-scale-mixture (resolvent) filtering',
        candidate='a(v)=1/(1+v/2), normalized rotary, 48 dimensions, no new trained parameters',
        primary_real='CE on already evaluated figurines+temple cameras, both raw RGB and frozen ResNet18; this is a post-hoc extension, not an untouched holdout',
        primary_point='visible R1 on fresh procedural seed namespace12100/12200/12300+seed; scale_noise primary robustness stress, clean oracle co-primary guardrail',
        gaussian_scale_control='select physical covariance multiplier .25/.5/1/2 by average training-seed Teatime validation CE (real) or validation loss separately by regime (point); never test selection',
        decision='Do not replace exponential PE unless both real feature branches retain clean CE and point scale_noise improves; report all alternatives and harms regardless',
        statistics='5 paired training seeds, query-camera clustered bootstrap for real; paired training-seed/scene bootstrap for procedural data',
        namespace_offset=10000 if args.domain == 'point' else None,
        point_query_note='No fitted query C in point task: q C=I/12; gaussian_none is both-fixed and equals old key-fixed behavior',
        limits='Known-scene sparse-track matching and rendered probes; no new-view rendering or real geometry-estimator uncertainty claim')
    (out/'frozen_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if args.domain == 'real':
        from experiments import region_pe_probe as runner
        runner.make_matcher = factory
    else:
        from experiments import gaussian_pe_probe as runner
        runner.make_matcher = factory
        original_dataset = runner.rendered_data.make_dataset
        def new_namespace(seed, *positional, **kw):
            return original_dataset(seed+10000, *positional, **kw)
        runner.rendered_data.make_dataset = new_namespace
    runner.main(SimpleNamespace(**config))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['real', 'point'], required=True)
    p.add_argument('--kind', choices=['raw', 'frozen'], default='raw')
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='11,22,33,44,55')
    p.add_argument('--methods', default='')
    main(p.parse_args())
