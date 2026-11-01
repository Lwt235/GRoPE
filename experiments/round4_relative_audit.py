"""Accuracy and CPU timing of fixed-query checkpoint reparameterization."""
import json
from pathlib import Path
import statistics
import time
import torch
from experiments.region_pe_probe import real_dataset
from experiments.query_support import use_fitted_query
from experiments.isometric_confirmation_analysis import sha256, write_json
from models.gaussian_spectral_pe import GaussianSpectralMatcher
from models.gaussian_relative_pe import RelativeSupportMatcher


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    out = Path('results/round4_relative_audit')
    out.mkdir(exist_ok=True)
    records = []
    benchmark_models = None
    for kind, dataset in [('raw', 'round3_fresh_real_data'), ('frozen', 'round3_fresh_frozen_visual_data')]:
        root = Path('results')/dataset
        datasets = []
        for scene in ['figurines', 'temple']:
            data, metadata = real_dataset(root, scene, 'test')
            data, _ = use_fitted_query(data, metadata, root/scene)
            datasets.append(data)
        for seed in [11, 22, 33, 44, 55]:
            checkpoint = Path(f'results/round4_spectral_real_{kind}/checkpoints/{seed}-gaussian_konly.pt')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            original = GaussianSpectralMatcher('gaussian_konly', 256, .25)
            original.load_state_dict(payload['state_dict'], strict=True)
            original.eval()
            converted = RelativeSupportMatcher(original).eval()
            logit_error = probability_error = 0.
            minimum_norm = float('inf')
            for data in datasets:
                for start in range(0, len(data['q']), 32):
                    part = {k: v[start:start+32] for k, v in data.items()}
                    for model in [original, converted]:
                        for side in ['q', 'k']:
                            minimum_norm = min(minimum_norm, float(model.encoder(part[side]).norm(dim=-1).min()))
                    a, b = original(part), converted(part)
                    logit_error = max(logit_error, float((a-b).abs().max()))
                    probability_error = max(probability_error, float((a.softmax(-1)-b.softmax(-1)).abs().max()))
            assert minimum_norm > 1e-6
            assert logit_error < 2e-5
            assert probability_error < 3e-6
            records.append(dict(kind=kind, seed=seed, checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint), maximum_logit_absolute_error=logit_error,
                maximum_probability_absolute_error=probability_error,
                minimum_original_or_converted_encoder_norm=minimum_norm,
                parameters=sum(p.numel() for p in converted.parameters())))
            if benchmark_models is None:
                benchmark_models = (original, converted)
                benchmark_data = {k: v[:16] for k, v in datasets[0].items()}
    timings = {name: [] for name in ['fixed_query_reference', 'relative_support']}
    for model in benchmark_models:
        for _ in range(20):
            model(benchmark_data)
    for repeat in range(7):
        for name, model in zip(timings, benchmark_models):
            start = time.perf_counter()
            for _ in range(100):
                model(benchmark_data)
            timings[name].append((time.perf_counter()-start)*10)
    medians = {name: statistics.median(value) for name, value in timings.items()}
    result = dict(records=records, full_matcher_cpu_milliseconds=medians,
        timing_repeats=timings, batch=16, query_and_key_tokens=48,
        speedup=medians['fixed_query_reference']/medians['relative_support'],
        scope='Exact real-arithmetic ordinary-domain conversion of fixed-query-covariance variant only; no new training or improvement over variable-query full model claimed. Dustbin input recovered by inverse constant filter.',
        limitations='CPU reference measurement, no GPU kernel claim; finite precision and F.normalize epsilon prohibit universal bitwise equivalence.',
        source_sha256={p: sha256(p) for p in ['models/gaussian_relative_pe.py',
            'models/gaussian_spectral_pe.py', 'models/gaussian_pe.py', __file__]})
    write_json(out/'summary.json', result)
    (out/'README.md').write_text('# Fixed-query relative support conversion\n\n'
        f"10 checkpoints, all original-partition figurines/temple pairs. Max logit error {max(r['maximum_logit_absolute_error'] for r in records):.3g}; "
        f"max probability error {max(r['maximum_probability_absolute_error'] for r in records):.3g}. "
        f"Minimum encoder norm {min(r['minimum_original_or_converted_encoder_norm'] for r in records):.3g}, above normalization epsilon.\n\n"
        f'CPU batch16, 48x48 pairs, seven repeats of100 forwards: {medians}; speedup {result["speedup"]:.3f}x. '
        'The model retains15171 learned parameters and48-dimensional Q/K. This is an optional simplification of gaussian_konly; '
        'removing query covariance from gaussian_full is a separate empirical change, not proved lossless. '
        'Fitted query means remain necessary. Training with Adam after parameter conversion need not follow the original optimization trajectory.\n')
    print(json.dumps(dict(timing=medians, speedup=result['speedup'],
        max_logit_error=max(r['maximum_logit_absolute_error'] for r in records))), flush=True)


if __name__ == '__main__':
    main()
