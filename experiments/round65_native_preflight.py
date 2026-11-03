"""Bounded fit-only native-input feasibility diagnostic after the R65 freeze.

This does not train/select task models or replace the complete frozen experiment.
"""
import argparse
import time
import numpy as np  # Required import order on this host.
import torch
from experiments.round65_common import OUT, PREP, RAW, CACHE_ROOT, CONFIG, ROLES, checked, read, write, sha
from experiments.round65_data import split_rows, read_vertices, native_arrays
from experiments.round65_run import configure, shared_digest
from models.round65_actual_gs import NativeGSClassifier

DEST = OUT/'native_preflight1'


def plan():
    checked()
    fit = split_rows(read(PREP/'train_catalog.json')['rows'])['fit']
    available = sorted((r for r in fit if (RAW/r['file_path']).is_file()), key=lambda r: r['file_path'])
    if len(available) < CONFIG['batch_size']:
        raise ValueError('not enough complete fitting files for the planned batch')
    DEST.mkdir(parents=True, exist_ok=False)
    selected = available[:CONFIG['batch_size']]
    write(DEST/'plan.json', dict(created_unix=time.time(), source_sha256=sha(__file__),
        lock_sha256=sha(OUT/'lock.json'), input_selection='lexicographically first16 currently complete frozen-fit entries',
        rows=selected, available_fit_files=len(available),
        hypothesis='fixed conversion and all five frozen models support finite native-input forward/backward within the existing memory ceiling',
        competing_explanation='native optimized scales cause ill-conditioning not represented by synthetic preflight inputs',
        fixed_comparison=dict(roles=ROLES, seed=11, batch=16, tokens=512, native_cache_tokens=2048,
            same_inputs=True, same_common_initialization=True, objective='mean squared random-initialization logits',
            optimizer_steps=0, labels_used=False),
        decisions=dict(pass_case='continue complete R65 unchanged; no task-efficacy claim',
            failure_case='preserve failure and localize conversion/model arithmetic; do not drop inputs or tune performance',
            floor_diagnostics='describe trace/axis contribution; no new pass/fail threshold and no post-hoc floor tuning'),
        limitations=['availability-selected diagnostic batch is not representative of all60 classes or both source domains',
                     'no task accuracy, training, validation, development evaluation or fresh confirmation'],
        numerical_arrays_read_at_plan=False, goal_complete=False))
    print('PLAN_CREATED', len(selected), 'fit files; no numerical arrays decoded', flush=True)


def quantiles(values):
    return np.quantile(values, [0, .5, .9, .99, 1]).tolist()


def run():
    checked(); specification = read(DEST/'plan.json')
    if sha(__file__) != specification['source_sha256'] or sha(OUT/'lock.json') != specification['lock_sha256']:
        raise ValueError('diagnostic or frozen experiment identity changed')
    if (DEST/'result.json').exists() or (DEST/'failure.json').exists():
        raise ValueError('preserve previous native-input diagnostic')
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError('native full-batch check requires an observed idle GPU')
    arrays = {k: [] for k in ('centers', 'covariance', 'appearance', 'indices')}
    conversion, axis_fractions, spectra, results = [], [], [], []
    try:
        for row in specification['rows']:
            path = RAW/row['file_path']
            if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
                raise ValueError('native diagnostic input identity mismatch '+row['file_path'])
            vertex = read_vertices(path)
            values, stats = native_arrays(vertex, row['file_path'], CONFIG['cache_tokens'])
            indices = values['indices']
            logs = np.column_stack([vertex[f'scale_{i}'][indices] for i in range(3)]).astype(np.float64)
            physical_axis_variance = np.exp(2*(logs-np.log(stats['rms_radius'])))
            axis_fraction = CONFIG['covariance_floor']/(physical_axis_variance+CONFIG['covariance_floor'])
            axis_fractions.append(axis_fraction.ravel()); spectra.append(physical_axis_variance.ravel())
            conversion.append(dict(path=row['file_path'], taxonomy_id=row['taxonomy_id'], **stats,
                physical_axis_variance_quantiles=quantiles(physical_axis_variance),
                axis_floor_fraction_quantiles=quantiles(axis_fraction)))
            for name in arrays:
                arrays[name].append(values[name][:CONFIG['tokens']])
            print('NATIVE_CONVERTED', row['file_path'], stats['vertices'], flush=True)
        stacked = {k: np.stack(v) for k, v in arrays.items()}
        cache = CACHE_ROOT/'native_preflight1'
        cache.mkdir(parents=True, exist_ok=False)
        np.savez(cache/'inputs.npz', **stacked)
        write(DEST/'conversion.json', dict(records=conversion, fit_only=True,
            diagnostic_input_path=str(cache/'inputs.npz'), diagnostic_input_sha256=sha(cache/'inputs.npz'),
            no_validation_or_development_evaluation=True))
        arguments = [torch.tensor(stacked[k], device='cuda') for k in ('centers', 'covariance', 'appearance')]
        common_identity = None
        for role in ROLES:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(11)
            model = NativeGSClassifier(role).cuda().train()
            identity = shared_digest(model)
            if common_identity is None:
                common_identity = identity
            if identity != common_identity:
                raise ValueError('native diagnostic common initialization mismatch')
            torch.manual_seed(65005)
            started = time.monotonic(); logits = model(*arguments)
            objective = logits.square().mean()
            objective.backward(); torch.cuda.synchronize()
            gradients = {n: p.grad for n, p in model.named_parameters()}
            if not torch.isfinite(logits).all() or any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError('nonfinite or missing native-input gradient '+role)
            peak = torch.cuda.max_memory_allocated()
            if peak > CONFIG['memory_ceiling_bytes']:
                raise MemoryError('existing allocation ceiling exceeded')
            positional_norms = {n: float(p.grad.norm()) for n, p in model.position.named_parameters()}
            results.append(dict(role=role, finite_logits_and_gradients=True, peak_allocated=peak,
                positional_gradient_norms=positional_norms, zero_gradient_parameters=[n for n, g in gradients.items() if not torch.count_nonzero(g)],
                shared_initial_sha256=identity, seconds=time.monotonic()-started))
            print('NATIVE_BACKWARD_OK', role, 'peak_bytes', peak, flush=True)
            del model, objective, logits, gradients
        fractions = np.concatenate(axis_fractions)
        write(DEST/'result.json', dict(passed=True, rows=results, gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__, source_sha256=sha(__file__), lock_sha256=sha(OUT/'lock.json'),
            plan_sha256=sha(DEST/'plan.json'), conversion_sha256=sha(DEST/'conversion.json'),
            physical_axis_variance_quantiles=quantiles(np.concatenate(spectra)),
            axis_floor_fraction_quantiles=quantiles(fractions), axis_fraction_above_half=float((fractions>.5).mean()),
            axis_fraction_above_nine_tenths=float((fractions>.9).mean()),
            trace_floor_median_of_object_medians=float(np.median([r['floor_fraction_quantiles'][1] for r in conversion])),
            observed_classes=sorted({r['taxonomy_id'] for r in specification['rows']}),
            classification='native-input feasibility only', task_models_trained=0,
            validation_or_dev_eval_access=False, fresh_confirmation=False, goal_complete=False))
        print('NATIVE_PREFLIGHT_COMPLETE; no task training or efficacy result', flush=True)
    except BaseException as error:
        write(DEST/'failure.json', dict(error_type=type(error).__name__, message=str(error),
            converted=[r['path'] for r in conversion], completed_roles=results,
            plan_sha256=sha(DEST/'plan.json'), goal_complete=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('plan', 'run'))
    (plan if parser.parse_args().action == 'plan' else run)()
