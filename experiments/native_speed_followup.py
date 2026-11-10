"""Exclusive speed checks and explicit private-instance training adaptation."""
import argparse
import os
import statistics as st
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.round65_common import ROOT, read, write, sha
from experiments import native_source_diversity as native
from experiments.efficiency_sidecar.check_cpu import old_inputs, variant, state_hash
from experiments.efficiency_sidecar.followup_gpu import base_model
from experiments.efficiency_sidecar.hidden_gpu import optimizer, step, cleanup
from experiments.efficiency_sidecar.fused_hidden_grad import fuse_hidden_gradient
from experiments.efficiency_sidecar.benchmark_gpu import resource_check

OUT = native.OUT / 'speed_transition'
IMPLEMENTATIONS = ('original', 'geometry_factored', 'geometry_fused')


def adapted(base, implementation):
    model = variant(base, 'original' if implementation == 'original' else 'geometry_factored')
    return fuse_hidden_gradient(model) if implementation == 'geometry_fused' else model


def paired_summary(rows, baseline, candidate):
    gains, aa, bb = [], [], []
    for block in range(8):
        cell = {r['label']: r for r in rows if r['block'] == block}
        for name in (baseline, candidate):
            assert cell[name+'_a']['final_loss'] == cell[name+'_b']['final_loss']
        a, b, c, d = [cell[k]['wall_ms'] for k in
                     (baseline+'_a', baseline+'_b', candidate+'_a', candidate+'_b')]
        gains.append(100*(1-(c+d)/(a+b)))
        aa.append(100*(1-b/a)); bb.append(100*(1-d/c))
    memory = max(r['peak_allocated_bytes'] for r in rows if r['label'].startswith(candidate+'_'))
    result = dict(baseline=baseline, candidate=candidate, paired_gain_percent=gains,
        median_gain_percent=st.median(gains), positive_blocks=sum(g > 0 for g in gains),
        baseline_drift_percent=aa, candidate_drift_percent=bb,
        candidate_peak_allocated_gib=memory/2**30)
    result['speed_supported'] = (result['median_gain_percent'] >= 3 and result['positive_blocks'] >= 7
        and result['median_gain_percent'] > 2*max(st.median(map(abs, aa)), st.median(map(abs, bb)))
        and memory <= native.CAP_BYTES)
    return result


def choose_implementation(comparisons, correct, stable):
    eligible = {name for name in IMPLEMENTATIONS[1:] if correct[name] and stable[name]
                and comparisons['original__'+name]['speed_supported']}
    if 'geometry_fused' in eligible and ('geometry_factored' not in eligible or
            comparisons['geometry_factored__geometry_fused']['speed_supported']):
        return 'geometry_fused'
    return 'geometry_factored' if 'geometry_factored' in eligible else 'original'


def sphere():
    out = OUT / 'sphere'
    assert not out.exists(), 'Preserve every earlier attempt.'
    native.check()
    out.mkdir(parents=True)
    started = time.time()
    record = dict(state='running', started_unix=started, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        mode='exclusive', role='fixed_isotropic5', batch=16, tokens=512, threads=4,
        resources=[], checks=[], measurements=[], stability=[],
        main_lock_sha256=sha(native.OUT/'lock.json'), source_sha256=sha(__file__),
        plan_sha256=sha(ROOT/'docs/native_speed_adoption_plan.md'))

    def resource(initialized=True):
        if time.time()-started > 600:
            raise TimeoutError('Ten minute stage limit; preserve partial measurements.')
        snapshot = resource_check(record['gpu_uuid'], initialized=initialized, shared_peers={})
        if not initialized:
            card = next(line.split(',') for line in snapshot['cards'].splitlines()
                        if line.split(',')[1].strip() == record['gpu_uuid'])
            assert int(card[2]) < 512 and int(card[4]) < 10, 'An idle GPU is required.'
        record['resources'].append(snapshot)

    try:
        resource(False)
        native.setup_cuda()
        torch.set_num_interop_threads(1)
        torch.ones(1, device='cuda').sum().item()
        resource()
        record.update(torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name())
        base, record['checkpoint'] = base_model('fixed_isotropic5')
        pool, target, record['input_identity'] = old_inputs(16, 2048)
        correct = {name: True for name in IMPLEMENTATIONS[1:]}
        for batch in (16, 3):
            inputs = tuple(t[:batch, :512].cuda() for t in pool)
            labels = target[:batch].cuda()
            states = []
            for name in IMPLEMENTATIONS:
                resource()
                model = adapted(base, name).cuda().train(); opt = optimizer(model)
                initial = state_hash(model)
                torch.manual_seed(831)
                logits = model(*inputs)
                loss = F.cross_entropy(logits, labels, label_smoothing=.1); loss.backward()
                gradients = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
                assert bool(torch.isfinite(gradients).all())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
                model.eval()
                with torch.inference_mode(): after = model(*inputs).cpu()
                states.append(dict(name=name, initial=initial, logits=logits.detach().cpu(), gradients=gradients,
                    parameters=torch.cat([p.detach().flatten().cpu() for p in model.parameters()]), after=after))
                del model, opt, logits, loss, gradients, after; cleanup()
            a = states[0]
            for b in states[1:]:
                assert a['initial'] == b['initial']
                error = dict(batch=batch, implementation=b['name'],
                    logit_max_abs=float((a['logits']-b['logits']).abs().max()),
                    gradient_relative_l2=float((a['gradients'].double()-b['gradients'].double()).norm()/a['gradients'].double().norm()),
                    parameter_update_max_abs=float((a['parameters']-b['parameters']).abs().max()),
                    after_update_logit_max_abs=float((a['after']-b['after']).abs().max()))
                error['passed'] = (error['logit_max_abs'] < 2e-4 and error['gradient_relative_l2'] < 1e-4
                                   and error['after_update_logit_max_abs'] < 2e-4)
                correct[b['name']] &= error['passed']
                record['checks'].append(error)
            del states, a, b, inputs, labels; cleanup()
        write(out/'correctness.json', record)
        rng = np.random.default_rng(925001)
        schedule = []
        while len(schedule) < 64:
            schedule.extend(native.runner.epoch_batches(rng, 16, batch_size=16, tokens=512))
        tags = tuple(name+'_'+rep for rep in ('a', 'b') for name in IMPLEMENTATIONS)
        for block in range(8):
            order = tags[block % 6:]+tags[:block % 6]
            for tag in order:
                resource()
                name = tag[:-2]
                model = adapted(base, name).cuda().train(); opt = optimizer(model)
                for j in range(2): step(model, opt, pool, target, schedule[j], 851000+20*block+j)
                model.zero_grad(set_to_none=True); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for j in range(2, 6): loss, norm = step(model, opt, pool, target, schedule[j], 851000+20*block+j)
                torch.cuda.synchronize()
                record['measurements'].append(dict(block=block, label=tag,
                    wall_ms=1000*(time.perf_counter()-start)/4, final_loss=float(loss), final_grad_norm=float(norm),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                del model, opt, loss, norm; cleanup()
            write(out/'progress.json', record)
            print('SPHERE_TIMING', block, flush=True)
        # Serial private trajectories avoid retaining three full training graphs.
        traces = {}
        for name in IMPLEMENTATIONS:
            model = adapted(base, name).cuda().train(); opt = optimizer(model)
            losses = []
            for i, spec in enumerate(schedule[:64]):
                if i % 8 == 0: resource()
                loss, norm = step(model, opt, pool, target, spec, 841000+i)
                losses.append(float(loss))
                assert np.isfinite(losses[-1]) and np.isfinite(float(norm))
            traces[name] = losses
            record['stability'].append(dict(implementation=name, losses=losses))
            del model, opt, loss, norm; cleanup()
            write(out/'stability_progress.json', record)
        stable = {name: max(abs(a-b) for a, b in zip(traces['original'], traces[name])) <= .02
                  for name in IMPLEMENTATIONS[1:]}
        comparisons = {a+'__'+b: paired_summary(record['measurements'], a, b) for a, b in
                      (('original', 'geometry_factored'), ('original', 'geometry_fused'),
                       ('geometry_factored', 'geometry_fused'))}
        decision = dict(implementation=choose_implementation(comparisons, correct, stable),
            comparisons=comparisons, correctness=correct, short_stability=stable,
            maximum_loss_difference={name: max(abs(a-b) for a, b in zip(traces['original'], traces[name]))
                                     for name in IMPLEMENTATIONS[1:]},
            complete_training_accuracy_equivalence_established=False,
            scientific_scores_used_for_decision=False)
        record.update(state='complete', elapsed_seconds=time.time()-started, decision=decision)
        resource()
        write(out/'result.json', record)
        decision.update(result_sha256=sha(out/'result.json'), plan_sha256=record['plan_sha256'])
        write(out/'decision.json', decision)
        print('SPHERE_RUNTIME_DECISION', decision['implementation'], flush=True)
    except BaseException as error:
        record.update(state='failed', error_type=type(error).__name__, reason=str(error))
        write(out/'failure.json', record)
        raise


def compile_stage(stage):
    from experiments.efficiency_sidecar import compile_gpu as module
    module.OUT = OUT / 'compile_exclusive'
    args = argparse.Namespace(stage=stage, mode='exclusive', batch=16, merge=False)
    record, out = module.initialize(args)
    try:
        getattr(module, stage)(record, out)
        module.resource(record)
        record.update(state='complete', elapsed_seconds=time.time()-record['started_unix'])
        write(out/(stage+'.json'), record)
    except BaseException as error:
        record.update(state='failed', error_type=type(error).__name__, error=str(error))
        write(out/(stage+'_failure.json'), record)
        raise


def train_pending(role, seed):
    manifest = read(OUT/'amendment.json')
    for p, digest in manifest['source_sha256'].items(): assert sha(p) == digest, p
    decision = read(OUT/'sphere/decision.json')
    assert sha(OUT/'sphere/result.json') == decision['result_sha256']
    assert [role, seed] in manifest['pending_at_hold'] and role == 'fixed_isotropic5'
    implementation = decision['implementation']
    previous = native.model_for

    def model_for(name):
        assert name == role
        return adapted(previous(name), implementation)

    native.model_for = model_for
    destination = native.OUT/'runs'/role/str(seed)
    receipt = dict(implementation=implementation, started_unix=time.time(),
        amendment_sha256=sha(OUT/'amendment.json'), decision_sha256=sha(OUT/'sphere/decision.json'),
        original_training_lock_sha256=sha(native.OUT/'lock.json'), source_sha256=manifest['source_sha256'])
    write(destination/f'execution_amendment_{time.time_ns()}.json', receipt)
    code = native.train(role, seed)
    if code == 0:
        run = read(destination/'run.json')
        run['execution_amendment'] = receipt
        write(destination/'run.json', run)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('sphere', 'compile_correctness', 'compile_timing', 'train'))
    parser.add_argument('--role'); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'sphere': sphere()
    elif args.action.startswith('compile_'): compile_stage(args.action.removeprefix('compile_'))
    else: raise SystemExit(train_pending(args.role, args.seed))
