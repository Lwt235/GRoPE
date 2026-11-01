"""Freeze the normalized Gaussian PE replication before matcher test evaluation."""
import hashlib
import json
from pathlib import Path

from experiments.isometric_confirmation_analysis import frozen_baselines
from experiments.rendered_probe import source_snapshot


OUT = Path('results/round3_normalized_confirmation')
METHODS = ['projected_rope', 'expected_fixed', 'expected_shape', 'isometric_shape',
           'normalized_fixed', 'normalized_shape', 'normalized_trace',
           'normalized_area', 'normalized_shuffled', 'absolute_geometry']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'frozen_protocol.json'
    if target.exists():
        raise FileExistsError('Never overwrite a frozen confirmation protocol')
    inputs = [Path('results/round3_fresh_real_data/selection_protocol.json')]
    for root in ['round3_fresh_real_data', 'round3_fresh_frozen_visual_data']:
        inputs.extend(sorted((Path('results')/root).glob('*/manifest.json')))
    for root in ['round3_real_support_data_adjacent', 'round3_frozen_visual_data']:
        inputs.append(Path('results')/root/'teatime/manifest.json')
    for kind in ['raw', 'frozen']:
        inputs.append(Path(f'results/round3_normalized_stable_{kind}_development/manifest.json'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    commands = {}
    for kind, root in [('raw', 'round3_real_support_data_adjacent'),
                       ('frozen', 'round3_frozen_visual_data')]:
        commands[kind] = ('python -m experiments.region_pe_probe --domain real '
            '--phase development --query-geometry fitted --scene teatime '
            f'--real-root results/{root} --out {OUT}/training_{kind} '
            '--seeds 11,22,33,44,55 --resolution .25 --steps 500 --val-every 20 '
            '--features 256 --batch 16 --methods '+','.join(METHODS))
    protocol = dict(
        stage='frozen before training or fresh-camera matcher evaluation',
        candidate='normalized_shape; Gaussian expected phase followed by per-token L2 Q/K normalization',
        mathematical_claim='48D, no additional learned parameters; nonlinear content-dependent map, not orthogonal',
        query='fitted projected query mean and covariance for every method; fixed/trace/area/shuffle intervene on key covariance only',
        source_hashes=source_snapshot(OUT), input_manifest_sha256=hashes,
        methods=METHODS, seeds=[11,22,33,44,55], training_commands=commands,
        training='Teatime train only; minimum Teatime validation CE selects checkpoint including step0; development runner is intentionally used to train without opening any test data',
        primary_data='previously unexported cameras, all four scenes; no target/prediction-based pair selection',
        primary_comparison='normalized_shape minus normalized_fixed CE on fresh figurines+temple (row-weighted); raw RGB primary, frozen ResNet18 features predeclared robustness replication',
        additional_comparisons='all controls, each scene and pooled four-scene CE/KL/JS/top1 target mass; same rows and candidate masks',
        inference='4000 paired crossed training-seed/query-camera bootstrap draws, scenes stratified; scene-macro means also reported to expose imbalance; per-seed differences reported',
        support_success='primary CE CI95 upper bound below zero and improvement in >=4/5 seed means; frozen-feature replication reported regardless of outcome',
        anisotropy_claim='requires additional paired CE CI95 upper bound below zero against both trace and area controls; otherwise attribute only general support benefit',
        overall_claim='compare projected RoPE, expected_fixed, isometric_shape, same-information descriptor MLP and frozen strong RBF; do not claim superior overall matching if strong controls win',
        secondary_data='already used Tea/Waldo test blocks: descriptive repeat only; never a new independent discovery claim',
        geometry_swap='both supplied disjoint-GS parity partitions, reported separately with counts; no favorable partition selection',
        limits='fixed known scenes; photos may have trained GS/SfM; conditional valid-support matching against sparse observed-track distributions, no dense occlusion/rejection or NVS claim',
        baseline=frozen_baselines(OUT))
    target.write_text(json.dumps(protocol, indent=2)+'\n')
    print(target, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
