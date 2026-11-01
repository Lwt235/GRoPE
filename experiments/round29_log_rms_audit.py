"""R29 independent canonical-log arithmetic, references and delivery audit.

Old20 bases retain the unchanged R27 numerical reference. Only the three
predeclared constant-depth diagonal bases receive a single analytic log-CDF
reference. No production density/preparation/normalization is imported.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp, ndtr
from experiments import round29_log_rms_contract as C
from experiments import round29_log_rms_fixtures as FIX
from experiments import round29_log_rms_reference as REF

ROOT=C.ROOT
DEFAULT_RUN=ROOT/'results/round29_log_rms_preflight'
DEFAULT_OUT=ROOT/'results/round29_log_rms_independent_audit'
APPROXIMATE_TIE_TOLERANCE=1e-12
SCOPE=dict(real_inputs_read=False,labels_read=False,general_numerical_accuracy_established=False,
           efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
sha,read,write,array_sha,inventory=C.sha,C.read,C.write,C.array_sha,C.inventory
source_names,check_runtime,inherited_bindings=C.source_names,C.check_runtime,C.inherited_bindings


def scope_check(value):
    if any(value.get(k) is not False for k in SCOPE):raise ValueError('all seven numerical-only scope flags must be explicit false')


def verify_inventory(directory):
    done=read(directory/'completion.json')
    C.verify_inventory(directory,done['files'],exclude=('completion.json',))
    return done

def verify_producer_lock(run):
    lock = read(run / 'lock.json'); fixtures = FIX.fixtures()
    scope_check(lock)
    if sha(run / 'lock.json') != (run / 'lock.sha256').read_text().strip():
        raise ValueError('producer lock changed')
    expected = dict(version=1, runtime=check_runtime(), policy=C.POLICY, reference_policy=C.REFERENCE_POLICY, gates=C.GATES,
        base_ids=[r['id'] for r in fixtures if r['kind'] == 'base'], identity_ids=[r['id'] for r in fixtures if r['kind'] == 'identity'],
        synthetic_only=True, **SCOPE)
    if any(lock.get(k) != value for k, value in expected.items()) or len(expected['base_ids']) != 23 or len(expected['identity_ids']) != 6:
        raise ValueError('fixed synthetic population/policy/runtime changed')
    if set(lock['source_hashes']) != set(source_names()):
        raise ValueError('joint transitive source closure differs')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT / name) != digest or sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('current/frozen source changed: ' + name)
    if (sha(run / 'fixtures.json') != lock['fixtures_sha256'] or read(run / 'fixtures.json') != fixtures
            or lock['bindings'] != inherited_bindings()):
        raise ValueError('fixed descriptors/provenance changed')
    return lock

def freeze(run, out):
    if out.exists() or any((run / name).exists() for name in ('started.json', 'records.json', 'arrays', 'completion.json')):
        raise ValueError('independent freeze must precede every formal case')
    lock = verify_producer_lock(run)
    if Path(lock['audit_root']).resolve() != out.resolve():
        raise ValueError('different declared audit directory')
    out.mkdir(parents=True)
    for name, digest in lock['source_hashes'].items():
        target = out / 'source_snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError('source changed while freezing independent snapshot')
    write(out / 'lock.json', dict(created_unix=time.time(), run=str(run.resolve()), producer_lock_sha256=sha(run / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], source_hashes=lock['source_hashes'], runtime=lock['runtime'],
        **SCOPE))
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')

def verify_joint_lock(run, out):
    lock, design = verify_producer_lock(run), read(out / 'lock.json')
    scope_check(design)
    expected = dict(run=str(run.resolve()), producer_lock_sha256=sha(run / 'lock.json'), fixtures_sha256=lock['fixtures_sha256'],
                    source_hashes=lock['source_hashes'], runtime=lock['runtime'], **SCOPE)
    if (sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip() or any(design.get(k) != v for k, v in expected.items())
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('independent joint lock differs')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('independent snapshot changed')
    return lock, design

def verify_seal(run, out):
    lock, design = verify_joint_lock(run, out)
    complete, seal = verify_inventory(run), read(run / 'forward_complete.json')
    records, fixtures = read(run / 'records.json'), read(run / 'fixtures.json')
    ids = lock['base_ids'] + lock['identity_ids']
    expected = dict(producer_lock_sha256=sha(run / 'lock.json'), audit_lock_sha256=sha(out / 'lock.json'),
        fixtures_sha256=lock['fixtures_sha256'], records_sha256=sha(run / 'records.json'), case_ids=ids, case_count=29,
        all_case_attempts_retained=True, all_successful_arrays_retained=True, synthetic_only=True, **SCOPE)
    if any(seal.get(k) != v for k, v in expected.items()) or [r['id'] for r in records] != ids or len(set(ids)) != 29:
        raise ValueError('complete ordered 29-case seal required')
    started = read(run / 'started.json')
    for stage in (started,seal,complete):scope_check(stage)
    if not math.isfinite(seal['phase_wall_seconds']) or seal['phase_wall_seconds']<0:
        raise ValueError('production phase duration must be finite nonnegative')
    if (started['producer_lock_sha256'] != sha(run / 'lock.json') or started['audit_lock_sha256'] != sha(out / 'lock.json')
            or not design['created_unix'] <= started['started_unix'] == seal['started_unix'] <= seal['phase_finished_unix'] <= seal['finished_unix']
            or complete['producer_forward_complete_sha256'] != sha(run / 'forward_complete.json')
            or complete['complete_case_inventory'] is not True
            or seal['phase_wall_passed'] != (seal['phase_wall_seconds'] <= lock['gates']['production_max_phase_wall_seconds'])
            or complete['phase_wall_passed'] != seal['phase_wall_passed']
            or seal['all_production_calls_completed'] != all(r['passed'] is True for r in records)
            or complete['all_production_calls_completed'] != seal['all_production_calls_completed']):
        raise ValueError('case chronology/status differs')
    expected_files = {'lock.json', 'lock.sha256', 'fixtures.json', 'started.json', 'records.json', 'forward_complete.json',
                      'call_starts.jsonl', 'model_results.jsonl'}
    expected_files |= {'source_snapshot/' + name for name in lock['source_hashes']} | {'arrays/' + name + '.npz' for name in ids}
    if set(complete['files']) != expected_files:
        raise ValueError('unsealed extra/missing production files')
    banks = {}
    starts=[json.loads(line) for line in (run/'call_starts.jsonl').read_text().splitlines()]
    journal=[json.loads(line) for line in (run/'model_results.jsonl').read_text().splitlines()]
    if ([r['id'] for r in journal]!=ids or [r['id'] for r in starts]!=[r['id'] for r in records if r['attempted']]
            or seal['attempted_calls']!=len(starts) or seal['all_available_partial_arrays_retained'] is not True):
        raise ValueError('production attempt/result journals do not match complete population')
    journal_by_id={r['id']:r for r in journal}; starts_by_id={r['id']:r for r in starts}
    for entry in starts+journal:scope_check(entry)
    for fixture, row in zip(fixtures, records):
        path = 'arrays/' + fixture['id'] + '.npz'
        if (row['archive_path'] != path or row['archive_sha256'] != sha(run / path) or row['kind'] != fixture['kind']
                or row['identity'] != fixture.get('identity') or row['policy_overrides'] != fixture['policy_overrides']
                or row['content_temperature'] != fixture['content_temperature']):
            raise ValueError('saved fixture identity changed')
        if row['bias_kind']!=fixture['bias_kind'] or type(row['attempted']) is not bool:
            raise ValueError('fixture bias or attempt metadata changed')
        if not started['started_unix']<=row['started_unix']<=row['finished_unix']<=seal['phase_finished_unix']:
            raise ValueError('per-call chronology differs')
        if row['attempted'] and not row['started_unix']<=starts_by_id[row['id']]['started_unix']<=row['finished_unix']:
            raise ValueError('actual model-start chronology differs')
        journal_entry=journal_by_id[row['id']]
        if journal_entry['available_array_sha256']!=row['array_sha256']:
            raise ValueError('journaled available arrays differ from retained archive')
        if any(row.get(k)!=v for k,v in journal_entry.items() if k!='available_array_sha256'):
            raise ValueError('journaled model outcome/work changed')
        with np.load(run / path, allow_pickle=False) as archive:
            bank = {name: archive[name] for name in archive.files}
        if {name: array_sha(value) for name, value in bank.items()} != row['array_sha256']:
            raise ValueError('saved array canonical identity changed')
        for name in ('mu', 'factor', 'bounds', 'content_bias'):
            if bank['input__' + name].dtype != np.float64 or not np.array_equal(bank['input__' + name], np.asarray(fixture[name], np.float64)):
                raise ValueError('input array differs from frozen descriptor')
        input_fields = {'input__' + name for name in ('mu', 'factor', 'bounds', 'content_bias')}
        success = {'log_raw_mass','log_raw_fov_mass','log_floored_mass','log_probability','probability',
                   'materialized_raw_indices','materialized_raw_mass','argmax'}
        partial = {'partial_log_raw_mass','partial_log_raw_fov_mass','partial_completed_cells',
                   'partial_completed_fov','partial_accumulated_tiles','partial_accumulated_fov_tiles'}
        if type(row.get('passed')) is not bool or row['status'] != ('complete' if row['passed'] else 'failed'):
            raise ValueError('explicit coherent status required')
        if row['passed']:
            if set(bank) != input_fields | success:
                raise ValueError('successful-array schema differs')
        elif not input_fields <= set(bank) <= input_fields | partial or not row.get('failure_reason'):
            raise ValueError('failed case must retain reason and allowed partial arrays only')
        for name in set(bank) - input_fields:
            if name == 'argmax':
                if bank[name].dtype != np.int32 or bank[name].shape != () or int(bank[name]) != row['argmax_index']:
                    raise ValueError('saved first-index argmax differs')
                continue
            if name in ('materialized_raw_indices','materialized_raw_mass'):
                dtype=np.int64 if name=='materialized_raw_indices' else np.float64
                if bank[name].dtype!=dtype or bank[name].ndim!=1:
                    raise ValueError('indexed ordinary materialization schema differs')
                continue
            shape = (1,) if name in ('log_raw_fov_mass','partial_log_raw_fov_mass','partial_completed_fov','partial_accumulated_fov_tiles') else (len(fixture['bounds']),)
            dtype=np.bool_ if name.startswith('partial_completed_') else (np.int64 if name.startswith('partial_accumulated_') else np.float64)
            if bank[name].dtype != dtype or bank[name].shape != shape:
                raise ValueError('output dtype/shape differs: ' + name)
        if (not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0 or type(row['peak_rss_kib_process']) is not int
                or row['peak_rss_kib_process'] < 0 or any(type(v) is not int or v < 0 for v in row['work'].values())):
            raise ValueError('retained work/time/RSS metadata invalid')
        scope_check(row)
        if not row['attempted'] and (row['passed'] or any(row['work'].values()) or set(bank)!=input_fields):
            raise ValueError('unattempted call must not invent results or work')
        if not row['passed']:partial_check(row,bank,dict(lock['policy'],**fixture['policy_overrides']))
        banks[fixture['id']] = bank
    return lock, seal, records, fixtures, banks

def structural_branch(mu, factor):
    rms = math.hypot(float(mu[2]), *map(float, factor[2]))
    if rms == 0: return 'zero_depth_rms', True
    if np.all(factor[2] == 0):
        return ('nonpositive_constant_depth', True) if mu[2] <= 0 else ('positive_constant_depth_gaussian', False)
    return 'variable_depth', False

def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()

def ordinary_probabilities(raw, bounds, bias, floor_density):
    raw, bounds, bias = map(np.asarray, (raw, bounds, bias))
    if (raw.dtype != np.float64 or raw.shape != (len(bounds),) or bias.dtype != np.float64 or bias.shape != raw.shape
            or not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(bias).all()):
        raise ValueError('finite nonnegative raw and native bias required')
    floor = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1) * floor_density
    if np.any(floor <= 0) or not np.isfinite(floor).all():
        raise ValueError('positive native area floor required')
    floored = raw + floor; logits = np.log(floored) + bias
    normalizer = float(logsumexp(logits))
    return np.exp(logits - normalizer), floor, floored, normalizer

def reference_check(result, mu, factor, bounds, bias, policy, reference_policy, gates, order):
    branch,structural_zero=structural_branch(mu,factor)
    rms=math.hypot(float(mu[2]),*map(float,factor[2]))
    m=0. if rms==0 else float(mu[2]/rms)
    sigma=0. if rms==0 else math.hypot(*map(float,factor[2]/rms))
    front=0. if structural_zero else (1. if sigma==0 else float(ndtr(m/sigma)))
    reference_branch={'zero_depth_rms':'zero_depth','nonpositive_constant_depth':'nonpositive_constant_depth',
        'positive_constant_depth_gaussian':'positive_constant_depth','variable_depth':'stochastic_depth'}[branch]
    work=result['work']
    metadata=(result['order']==order and result['structural_branch']==reference_branch
        and result['structural_zero'] is structural_zero
        and result['method']=='independent-SVD-homogeneous-depth-Gaussian-rectangle'
        and np.isclose(result['front_mass'],front,rtol=2e-12,atol=0)
        and np.isclose(result['rms_depth'],rms,rtol=2e-15,atol=0)
        and np.isclose(result['normalized_depth_mean'],m,rtol=2e-15,atol=0)
        and np.isclose(result['normalized_depth_std'],sigma,rtol=2e-15,atol=0)
        and all(type(v) is int and v>=0 for v in work.values())
        and work['cell_node_evaluations']==work['elementary_inner_nodes']==work['inner_x_nodes']+work['narrow_y_pdf_nodes']
        and work['cell_node_evaluations']<=reference_policy['max_call_cell_node_evaluations']
        and work['depth_evaluations']<=reference_policy['maximum_depth_evaluations']
        and work['outer_intervals']<=reference_policy['maximum_depth_intervals'])
    raw=result['raw_mass']; probability,floor,floored,_=ordinary_probabilities(raw,bounds,bias,policy['floor_density'])
    total=math.fsum(map(float,raw)); norm=abs(math.fsum(map(float,probability))-1.)
    direct=result['raw_fov_mass'];direct_error=abs(direct-total)
    positive=(np.all(raw==0) and direct==0) if structural_zero else (np.all(raw>0) and math.isfinite(direct) and direct>0)
    arithmetic=float(np.max(abs(probability-result['probability'])))
    tail=result['truncation_bounds']
    if structural_zero:depth_tail=inner_tail=0.
    else:
        cutoff=reference_policy['depth_cutoff']
        threshold=-m/sigma if sigma else None
        depth_tail=0. if sigma==0 else (float(ndtr(-cutoff)-ndtr(threshold)) if threshold < -cutoff else 0.)+float(ndtr(-max(cutoff,threshold)))
        inner_tail=float(2.*ndtr(-reference_policy['inner_x_cutoff']))*front
    expected_total=depth_tail+inner_tail
    expected_p=0. if structural_zero else min(1.,2.*expected_total/math.fsum(map(float,floor*np.exp(bias-bias.max()))))
    tail_metadata=all(np.isclose(tail[k],v,rtol=2e-12,atol=0) for k,v in dict(
        depth_omitted_mass=depth_tail,inner_x_omitted_mass=inner_tail,native_raw_l1_absolute=expected_total,
        direct_fov_absolute=expected_total,probability_max_absolute=expected_p).items())
    if structural_zero:
        tails=(tail['native_raw_l1_absolute']==0 and tail['direct_fov_absolute']==0
               and tail['raw_l1_over_reference_fov'] is None and tail['probability_max_absolute']==0)
    else:
        if total<=0 or not math.isfinite(total):raise ValueError('positive reference raw FOV is unresolved')
        tails=(tail['native_raw_l1_absolute']>=0 and tail['direct_fov_absolute']>=0
               and np.isclose(tail['raw_l1_over_reference_fov'],tail['native_raw_l1_absolute']/total,rtol=1e-12,atol=0)
               and 0<=tail['raw_l1_over_reference_fov']<=gates['truncation_raw_l1_relative']
               and 0<=tail['probability_max_absolute']<=gates['truncation_probability_absolute'])
    passed=bool(result.get('passed') is True and positive and tails and tail_metadata and metadata
        and norm<=gates['normalization_absolute'] and arithmetic<=gates['normalization_absolute']
        and direct_error<=gates['direct_fov_additivity_absolute'] and np.array_equal(floored,result['floored_mass']))
    return dict(passed=passed,normalization_error=norm,same_raw_probability_error=arithmetic,
        direct_fov_additivity_error=direct_error,truncation_gates_passed=bool(tails),
        metadata_consistent=bool(metadata),truncation_metadata_consistent=bool(tail_metadata),
        reference_positive_or_structural_zero=bool(positive),raw_fov_mass_sum=total)


def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def exp_diagnostic(value):
    if value==-np.inf:return 0.
    if value>math.log(np.finfo(np.float64).max):return None
    return math.exp(float(value))


def log_absolute_difference(a,b):
    """Log |exp(a)-exp(b)| without ordinary raw materialization."""
    a,b=np.broadcast_arrays(np.asarray(a,np.float64),np.asarray(b,np.float64))
    if np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any():
        raise ValueError('canonical log masses may be finite or exact negative infinity only')
    high=np.maximum(a,b);result=np.full(a.shape,-np.inf,np.float64)
    different=a!=b
    result[different]=high[different]+np.log(-np.expm1(-np.abs(a[different]-b[different])))
    return result


def validate_logs(log_raw,structural_zero):
    log_raw=np.asarray(log_raw)
    if log_raw.dtype!=np.float64 or log_raw.ndim!=1 or not len(log_raw):
        raise ValueError('nonempty float64 canonical native log mass required')
    if structural_zero:
        if not np.isneginf(log_raw).all():raise ValueError('structural zero must have exact -inf log mass')
    elif not np.isfinite(log_raw).all():
        raise ValueError('positive nonstructural raw mass must have finite canonical logs')
    return log_raw


def compare_log_raw(actual,reference,bounds,bias,floor_density,gates,*,reference_comparison=False,
                    archived_probability=None,structural_zero=False):
    actual=validate_logs(actual,structural_zero);reference=validate_logs(reference,structural_zero)
    if actual.shape!=reference.shape:raise ValueError('native log mass shapes differ')
    pa=REF.probabilities(actual,bounds,bias,floor_density)['probability']
    pr=REF.probabilities(reference,bounds,bias,floor_density)['probability']
    if archived_probability is not None:
        pa=np.asarray(archived_probability)
        if pa.dtype!=np.float64 or pa.shape!=actual.shape or not np.isfinite(pa).all() or np.any(pa<0):
            raise ValueError('finite archived native probability required')
    log_a=float(logsumexp(actual));log_r=float(logsumexp(reference))
    log_diff=log_absolute_difference(actual,reference)
    if structural_zero:log_fov_relative=log_l1_relative=-np.inf
    else:
        log_fov_relative=float(log_absolute_difference(log_a,log_r))-log_r
        log_l1_relative=float(logsumexp(log_diff))-log_r
    relative_gate=gates['reference_raw_relative'] if reference_comparison else gates['raw_fov_relative']
    l1_gate=gates['reference_raw_relative'] if reference_comparison else gates['raw_l1_over_reference_fov']
    p_gate=gates['reference_probability_max_absolute'] if reference_comparison else gates['probability_max_absolute']
    pe=float(np.max(abs(pa-pr)))
    log_cell_relative=log_diff-reference if not structural_zero else np.full(actual.shape,-np.inf)
    return dict(passed=bool((structural_zero or (log_fov_relative<=math.log(relative_gate) and log_l1_relative<=math.log(l1_gate))) and pe<=p_gate),
        structural_zero=structural_zero,raw_relative_applicable=not structural_zero,
        log_raw_fov_actual=finite_or_none(log_a),log_raw_fov_reference=finite_or_none(log_r),
        raw_fov_actual_materialized_diagnostic=exp_diagnostic(log_a),raw_fov_reference_materialized_diagnostic=exp_diagnostic(log_r),
        raw_fov_relative=None if structural_zero else exp_diagnostic(log_fov_relative),
        raw_l1_over_reference_fov=None if structural_zero else exp_diagnostic(log_l1_relative),
        log_raw_fov_relative=finite_or_none(log_fov_relative),log_raw_l1_over_reference_fov=finite_or_none(log_l1_relative),
        maximum_probability_error=pe,per_cell_log_absolute_error=[finite_or_none(x) for x in log_diff],
        per_cell_relative_error=[None if structural_zero else exp_diagnostic(x) for x in log_cell_relative],
        per_cell_log_relative_error=[finite_or_none(x) for x in log_cell_relative],
        zero_reference_cells=len(reference) if structural_zero else 0,actual_argmax=int(pa.argmax()),reference_argmax=int(pr.argmax()),
        argmax_changed=bool(pa.argmax()!=pr.argmax()),actual_exact_maximizers=np.flatnonzero(pa==pa.max()).tolist(),
        reference_exact_maximizers=np.flatnonzero(pr==pr.max()).tolist(),
        actual_approximate_maximizers=np.flatnonzero(abs(pa-pa.max())<=APPROXIMATE_TIE_TOLERANCE).tolist(),
        reference_approximate_maximizers=np.flatnonzero(abs(pr-pr.max())<=APPROXIMATE_TIE_TOLERANCE).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE)


def direct_fov_check(log_raw,log_direct,structural_zero,absolute_gate):
    raw=validate_logs(log_raw,structural_zero);direct=np.asarray(log_direct)
    if direct.dtype!=np.float64 or direct.shape!=(1,):raise ValueError('one canonical direct FOV log required')
    validate_logs(direct,structural_zero)
    error=float(log_absolute_difference(float(logsumexp(raw)),float(direct[0])))
    return dict(passed=bool(error<=math.log(absolute_gate)),log_absolute_error=finite_or_none(error),
        absolute_error_materialized_diagnostic=exp_diagnostic(error))


def materialization_check(bank,structural_zero):
    raw=validate_logs(bank['log_raw_mass'],structural_zero)
    with np.errstate(under='ignore',over='ignore'):ordinary=np.exp(raw)
    if not np.isfinite(ordinary).all() or not np.isfinite(bank['materialized_raw_mass']).all():
        raise ValueError('ordinary materialization overflow is a failed admission, never valid packed output')
    indices=np.arange(len(raw),dtype=np.int64) if structural_zero else np.flatnonzero(ordinary>0)
    values=np.zeros(len(raw),np.float64) if structural_zero else ordinary[indices]
    if (bank['materialized_raw_indices'].dtype!=np.int64 or bank['materialized_raw_mass'].dtype!=np.float64
            or not np.array_equal(bank['materialized_raw_indices'],indices)
            or not np.array_equal(bank['materialized_raw_mass'],values)):
        raise ValueError('indexed ordinary materialization must retain exactly exp(lograw)>0, or structural zeros')
    return dict(ordinary_raw_unmaterialized_count=0 if structural_zero else len(raw)-len(indices),
        ordinary_raw_subnormal_count=int(np.count_nonzero((values>0)&(values<np.finfo(np.float64).tiny))))


def identity_check(fixture,actual,parent,gates):
    declaration=fixture['identity'];relation=declaration['relation']
    expected={n:parent[n] for n in ('log_raw_mass','log_floored_mass','probability')}
    if relation=='permutation':expected={n:v[declaration['permutation']] for n,v in expected.items()}
    elif relation=='merge':
        expected={n:np.asarray([float(logsumexp(v[g])) if n!='probability' else math.fsum(map(float,v[g]))
                  for g in declaration['groups']],np.float64) for n,v in expected.items()}
    elif relation!='equal':raise ValueError('unfrozen identity relation')
    logs={n:float(np.max(log_absolute_difference(actual[n],expected[n]))) for n in ('log_raw_mass','log_floored_mass')}
    pe=float(np.max(abs(actual['probability']-expected['probability'])))
    return dict(id=fixture['id'],parent=declaration['parent'],relation=relation,
        passed=bool(pe<=gates['integrated_identity_absolute'] and all(v<=math.log(gates['integrated_identity_absolute']) for v in logs.values())),
        probability_absolute_error=pe,**{n+'_log_absolute_error':finite_or_none(v) for n,v in logs.items()},
        **{n+'_absolute_error_materialized_diagnostic':exp_diagnostic(v) for n,v in logs.items()})


def near_constant_check(banks,production,gates):
    near,constant='11_near_constant','03_constant_correlated'
    if not production[near]['passed'] or not production[constant]['passed']:return dict(passed=False,failure_reason='near/constant producer failed')
    a,b=banks[near],banks[constant]
    log_relative=float(logsumexp(log_absolute_difference(a['log_raw_mass'],b['log_raw_mass'])))-float(logsumexp(b['log_raw_mass']))
    pe=float(np.max(abs(a['probability']-b['probability'])))
    branch=production[near]['structural_branch']=='variable_depth' and np.any(a['input__factor'][2]!=0)
    return dict(passed=bool(branch and pe<=gates['near_constant_limit_absolute'] and log_relative<=math.log(gates['near_constant_limit_raw_relative'])),
        probability_max_absolute=pe,raw_l1_over_constant_fov=exp_diagnostic(log_relative),
        log_raw_l1_over_constant_fov=finite_or_none(log_relative),nonzero_depth_branch_preserved=bool(branch))


LSE_SCOPES=('tile_nodes','cell_tiles','fov_tiles','native_total','floor','floor_total','floored_total','content_normalization')


def local_lse_diagnostic(terms):
    """Reconstruct local-loss metadata for saved log inputs, not density nodes."""
    terms=np.asarray(terms,np.float64)
    if terms.ndim==1:terms=terms[None,:]
    maximum=np.max(terms,axis=-1,keepdims=True);finite=np.isfinite(terms)
    with np.errstate(invalid='ignore',under='ignore'):
        shifted=np.where(finite,terms-maximum,-np.inf);scaled=np.exp(shifted)
    lost=finite&(scaled==0);counts=np.sum(lost,axis=-1);affected=counts>0
    result=dict(sum_count=len(terms),underflow_sum_count=int(affected.sum()),underflow_term_count=int(counts.sum()),
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None)
    if np.any(affected):
        biggest=np.max(np.where(lost,shifted,-np.inf),axis=-1)[affected]
        bound=np.log(counts[affected])+biggest
        retained=np.asarray([math.fsum(map(float,row)) for row in scaled[affected]])
        result.update(maximum_log_scaled_omitted_bound=float(bound.max()),
            maximum_log_absolute_omitted_bound=float((maximum[:,0][affected]+bound).max()),
            maximum_log_relative_omitted_bound=float((bound-np.log(retained)).max()))
    return result


def lse_check(row,bank,zero,native_tiles,fov_tiles,tolerance):
    report=row['lse_diagnostics'];scopes=report['by_scope']
    if (set(scopes)!=set(LSE_SCOPES) or report['exact_arithmetic_local_omission_only'] is not True
            or report['is_pipeline_error_bound'] is not False
            or report['excludes_addition_absorption_density_quadrature_and_roundoff'] is not True):
        raise ValueError('LSE diagnostics must retain local-only scope')
    n=len(bank['log_raw_mass']);counts=dict(tile_nodes=native_tiles+fov_tiles,cell_tiles=native_tiles,fov_tiles=fov_tiles,
        native_total=1,floor=n,floor_total=1,floored_total=1,content_normalization=1)
    sizes=dict(tile_nodes=576,cell_tiles=2,fov_tiles=2,native_total=n,floor=2,floor_total=n,floored_total=n,content_normalization=n)
    fields={'sum_count','underflow_sum_count','underflow_term_count','maximum_log_scaled_omitted_bound',
            'maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'}
    for name,r in scopes.items():
        if set(r)!=fields or r['sum_count']!=counts[name] or any(type(r[k]) is not int or r[k]<0 for k in ('sum_count','underflow_sum_count','underflow_term_count')):
            raise ValueError('local positive sum counts differ')
        if not (0<=r['underflow_sum_count']<=r['sum_count'] and r['underflow_sum_count']<=r['underflow_term_count']<=r['underflow_sum_count']*(sizes[name]-1)):
            raise ValueError('impossible positive-sum underflow counts')
        bounds=[r[k] for k in ('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound')]
        if not r['underflow_term_count']:
            if any(v is not None for v in bounds):raise ValueError('no lost terms requires null local bounds')
        elif not all(isinstance(v,(float,int)) and math.isfinite(v) for v in bounds):
            raise ValueError('lost positive terms require finite local log bounds')
        elif bounds[2]>bounds[0]+tolerance or bounds[0]>math.log(max(1,sizes[name]-1))+math.log(np.nextafter(0.,1.))+tolerance:
            raise ValueError('local omitted bound incompatible with exp-underflow event')
    floor=np.prod(bank['input__bounds'][:,2:]-bank['input__bounds'][:,:2],axis=1)*row['floor_density']
    logfloor=np.log(floor)
    saved_inputs=dict(native_total=bank['log_raw_mass'],floor=np.column_stack((bank['log_raw_mass'],logfloor)),
        floor_total=logfloor,floored_total=bank['log_floored_mass'],
        content_normalization=bank['log_floored_mass']+bank['input__content_bias'])
    for name,terms in saved_inputs.items():
        expected=local_lse_diagnostic(terms)
        for key,value in expected.items():
            actual=scopes[name][key]
            if value is None or isinstance(value,int):
                if actual!=value:raise ValueError('saved-input LSE local loss diagnostic differs: '+name+'/'+key)
            elif actual is None or abs(actual-value)>tolerance:
                raise ValueError('saved-input LSE local log bound differs: '+name+'/'+key)
    return dict(passed=True,saved_input_scopes_independently_recomputed=list(saved_inputs),
        node_and_tile_scopes_not_numerically_replayed=['tile_nodes','cell_tiles','fov_tiles'],
        local_omitted_bound_is_pipeline_certificate=False)


def production_check(row,bank,policy,gates):
    if row['passed'] is not True:return dict(passed=False,status='production_failed',failure_reason=row.get('failure_reason'))
    mu,factor,bounds,bias=[bank['input__'+n] for n in ('mu','factor','bounds','content_bias')]
    branch,zero=structural_branch(mu,factor);raw=validate_logs(bank['log_raw_mass'],zero)
    expected=REF.probabilities(raw,bounds,bias,policy['floor_density'])
    p=bank['probability'];logp=bank['log_probability'];lf=bank['log_floored_mass']
    if (not np.isfinite(p).all() or np.any(p<0) or not np.isfinite(logp).all() or not np.isfinite(lf).all()):
        raise ValueError('finite completed log floor, log probability and ordinary probability required')
    rms=math.hypot(float(mu[2]),*map(float,factor[2]));m=None if rms==0 else float(mu[2]/rms)
    sigma=None if rms==0 else math.hypot(*map(float,factor[2]/rms))
    front=0. if zero else (1. if sigma==0 else float(ndtr(m/sigma)))
    fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)];maxside=policy['h']*policy['maximum_tile_side_in_h']
    native=0 if zero else sum(int(np.prod(np.ceil((box[2:]-box[:2])/maxside).astype(np.int64))) for box in bounds)
    fv=0 if zero else int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside).astype(np.int64)))
    work=dict(cell_node_evaluations=(native+fv)*24**2,native_tiles_completed=native,fov_tiles_completed=fv)
    material=materialization_check(bank,zero);direct=direct_fov_check(raw,bank['log_raw_fov_mass'],zero,gates['direct_fov_additivity_absolute'])
    diag=row['density_diagnostics'];series_keys=('maximum_log_negative_tail_relative_remainder','minimum_scaled_negative_tail_series_lower_bound','maximum_scaled_negative_tail_series_upper_bound')
    nodekeys=('constant_depth_nodes','positive_branch_nodes','negative_branch_nodes','negative_erfcx_nodes','negative_series_nodes')
    if any(type(diag[k]) is not int or diag[k]<0 for k in nodekeys):raise ValueError('density branch counts invalid')
    if diag['negative_series_nodes']:
        rel,lower,upper=[diag[k] for k in series_keys]
        series=all(math.isfinite(v) for v in (rel,lower,upper)) and 0<lower<=upper and rel<=math.log(policy['max_negative_tail_relative_remainder'])
    else:series=all(diag[k] is None for k in series_keys)
    density_range=(diag['maximum_log_density'] is None and diag['minimum_log_density'] is None) if zero else (
        math.isfinite(diag['maximum_log_density']) and math.isfinite(diag['minimum_log_density']) and diag['minimum_log_density']<=diag['maximum_log_density'])
    bound=0. if zero else float(np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum())/(math.pi*policy['h']**2*math.comb(48,24))
    trunc=row['analytic_truncation_bound'];tol=gates['normalization_absolute']
    logsum=None if zero else float(logsumexp(raw));logfloor=float(logsumexp(expected['log_floor_mass']))
    metadata=bool(row['structural_branch']==branch and row['structural_zero'] is zero and series and density_range
        and np.isclose(row['front_mass'],front,rtol=2e-13,atol=0) and row['rms_depth']==rms
        and row['normalized_depth_mean']==m and row['normalized_depth_std']==sigma
        and row['quadrature_order']==24 and row['maximum_tile_side_in_h']==2. and row['h']==policy['h'] and row['floor_density']==policy['floor_density']
        and row['projection_regularization']=='rms_depth_xy_gaussian' and row['canonical_raw_representation']=='log_mass'
        and row['ordinary_materialization_is_canonical'] is False and row['tail_truncation_used'] is False
        and row['quadrature_error_bound_available'] is False and row['exact_arithmetic_gl_bound_available'] is True
        and row['native_tile_count']==native and row['fov_tile_count']==fv and row['work']==work
        and trunc['exact_arithmetic_only'] is True and trunc['excludes_density_evaluation_and_roundoff'] is True
        and trunc['is_relative_error_certificate'] is False and trunc['raw_native_l1_absolute']==bound and trunc['raw_direct_fov_absolute']==bound
        and diag['constant_depth_nodes']+diag['positive_branch_nodes']+diag['negative_branch_nodes']==work['cell_node_evaluations']
        and diag['negative_erfcx_nodes']+diag['negative_series_nodes']==diag['negative_branch_nodes']
        and (row['log_raw_mass_sum'] is None if zero else (isinstance(row['log_raw_mass_sum'],(float,int)) and abs(row['log_raw_mass_sum']-logsum)<=tol))
        and abs(row['log_total_area_floor']-logfloor)<=tol and abs(row['log_normalizer']-float(logsumexp(lf)))<=tol
        and abs(row['log_content_normalizer']-expected['log_content_normalizer'])<=tol
        and row['ordinary_raw_unmaterialized_count']==material['ordinary_raw_unmaterialized_count']
        and row['ordinary_raw_subnormal_count']==material['ordinary_raw_subnormal_count']
        and row['zero_raw_cells']==(len(raw) if zero else 0) and row['all_raw_mass_zero'] is zero
        and row['area_prior_used'] is bool(zero and np.all(bias==bias[0]))
        and row['raw_mass_below_total_floor'] is bool(zero or logsum<logfloor)
        and row['argmax_index']==int(p.argmax()) and row['argmax_exact_ties']==int(np.count_nonzero(p==p.max()))
        and row['argmax_near_ties_at_1e_12']==int(np.count_nonzero(p.max()-p<=1e-12)))
    local=lse_check(row,bank,zero,native,fv,tol)
    norm=abs(math.fsum(map(float,p))-1.);pe=float(np.max(abs(p-expected['probability'])))
    lfe=float(np.max(abs(lf-expected['log_floored_mass'])));lpe=float(np.max(abs(logp-expected['log_probability'])))
    with np.errstate(under='ignore'):saved_exp_error=float(np.max(abs(np.exp(logp)-p)))
    passed=bool(metadata and direct['passed'] and local['passed'] and work['cell_node_evaluations']<=policy['max_cell_node_evaluations']
        and max(norm,pe,lfe,lpe,saved_exp_error)<=tol)
    return dict(passed=passed,status='checked',metadata_consistent=metadata,structural_branch=branch,structural_zero=zero,
        normalization_error=norm,same_raw_probability_error=pe,log_floor_arithmetic_error=lfe,log_probability_arithmetic_error=lpe,
        saved_probability_exp_error=saved_exp_error,direct_fov=direct,lse_diagnostic_check=local,**material,
        log_raw_mass_sum=logsum,log_total_area_floor=logfloor,raw_below_total_floor=bool(zero or logsum<logfloor),
        all_raw_mass_zero=zero,exact_maximizers=np.flatnonzero(p==p.max()).tolist(),work_expected=work)


def partial_check(row,bank,policy):
    names={n for n in bank if not n.startswith('input__')}
    expected={'partial_log_raw_mass','partial_log_raw_fov_mass','partial_completed_cells','partial_completed_fov',
              'partial_accumulated_tiles','partial_accumulated_fov_tiles'}
    if not names:return dict(passed=True,available_partial_arrays=[])
    if names!=expected:raise ValueError('returned partial log integration requires all masks and tile counts')
    for logkey,countkey,maskkey in (('partial_log_raw_mass','partial_accumulated_tiles','partial_completed_cells'),
                                   ('partial_log_raw_fov_mass','partial_accumulated_fov_tiles','partial_completed_fov')):
        logs,counts,mask=bank[logkey],bank[countkey],bank[maskkey]
        if np.isnan(logs).any() or np.isposinf(logs).any() or np.any(counts<0):raise ValueError('invalid partial log sum/count')
        if not np.array_equal(np.isfinite(logs),counts>0):raise ValueError('partial finite sums must distinguish accumulated from uncomputed cells')
        if np.any(mask&(counts==0)) and not row.get('structural_zero',False):raise ValueError('positive completed cell needs accumulated tiles')
    if (int(bank['partial_accumulated_tiles'].sum())!=row['work']['native_tiles_completed']
            or int(bank['partial_accumulated_fov_tiles'].sum())!=row['work']['fov_tiles_completed']
            or row['work']['cell_node_evaluations']>policy['max_cell_node_evaluations']):
        raise ValueError('partial tile work differs or exceeds fixed budget')
    zero=structural_branch(bank['input__mu'],bank['input__factor'])[1]
    if zero:
        all_masks=np.r_[bank['partial_completed_cells'],bank['partial_completed_fov']]
        if (any(np.any(bank[n]) for n in ('partial_accumulated_tiles','partial_accumulated_fov_tiles'))
                or not (np.all(all_masks) or not np.any(all_masks))):
            raise ValueError('structural-zero partial state must be wholly before or after its exact branch')
    else:
        bounds=bank['input__bounds'];maxside=policy['h']*policy['maximum_tile_side_in_h']
        totals=np.asarray([int(np.prod(np.ceil((box[2:]-box[:2])/maxside))) for box in bounds],np.int64)
        fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)]
        ftotal=np.asarray([int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside)))],np.int64)
        for counts,masks,expected in ((bank['partial_accumulated_tiles'],bank['partial_completed_cells'],totals),
                (bank['partial_accumulated_fov_tiles'],bank['partial_completed_fov'],ftotal)):
            if np.any(counts>expected) or not np.array_equal(masks,counts==expected):
                raise ValueError('partial completion masks must match independently derived full tile counts')
    return dict(passed=True,available_partial_arrays=sorted(names),
        completed_native_cells=int(bank['partial_completed_cells'].sum()),
        finite_partial_native_cells=int(np.isfinite(bank['partial_log_raw_mass']).sum()),
        incomplete_finite_native_cells=int(np.count_nonzero(np.isfinite(bank['partial_log_raw_mass'])&~bank['partial_completed_cells'])))


def analytic_check(result,bank,policy,gates):
    n=len(bank['input__bounds']);logs=validate_logs(result['log_raw_mass'],False)
    expected=REF.probabilities(logs,bank['input__bounds'],bank['input__content_bias'],policy['floor_density'])
    direct=direct_fov_check(logs,result['log_raw_fov_mass'],False,gates['direct_fov_additivity_absolute'])
    norm=abs(math.fsum(map(float,result['probability']))-1.)
    error=max(float(np.max(abs(result[k]-expected[k]))) for k in ('log_floored_mass','log_probability','probability'))
    metadata=bool(result['passed'] is True and result['method']=='independent-constant-depth-diagonal-normal-logCDF-product'
        and result['reference_route']=='analytic_single_call' and result['order'] is None
        and result['tail_truncation_used'] is False and result['quadrature_error_bound_available'] is False
        and result['work']==dict(cell_node_evaluations=0,analytic_cdf_evaluations=4*(n+1))
        and result['analytic_cdf_evaluations']==4*(n+1) and result['structural_zero'] is False
        and np.array_equal(result['raw_structural_zero'],np.zeros(n,np.bool_)))
    return dict(passed=bool(metadata and direct['passed'] and max(norm,error)<=gates['normalization_absolute']),
        metadata_consistent=metadata,normalization_error=norm,same_raw_arithmetic_error=error,direct_fov=direct,
        analytic_special_case_only=True,order_refinement_performed=False,floating_CDF_error_bound_available=False)


def audit(run,out):
    if any((out/n).exists() for n in ('audit_started.json','completion.json','summary.json')):
        raise ValueError('preserve every prior formal audit attempt')
    started,phase=time.time(),time.perf_counter();write(out/'audit_started.json',dict(started_unix=started,**SCOPE))
    production=[];references=[];comparisons=[];identities=[]
    try:
        lock,seal,records,fixtures,banks=verify_seal(run,out)
        if started<seal['finished_unix']:raise ValueError('references must follow complete global production seal')
        fmap={f['id']:f for f in fixtures}
        for fixture,row in zip(fixtures,records):
            policy=dict(lock['policy'],**fixture['policy_overrides'])
            try:
                check=production_check(row,banks[fixture['id']],policy,lock['gates'])
                if not row['passed']:check['partial_delivery']=partial_check(row,banks[fixture['id']],policy)
            except Exception as error:check=dict(passed=False,status='saved_output_failure',failure_reason=type(error).__name__+': '+str(error))
            production.append(dict(id=fixture['id'],**check,**SCOPE));write(out/'production_checks.json',production)
        pmap={r['id']:r for r in production};budget=REF.NodeBudget(lock['reference_policy']['max_total_cell_node_evaluations'])
        cumulative=cdf_count=0;directory=out/'reference_arrays';directory.mkdir()
        for name in ('reference_starts.jsonl','reference_results.jsonl'):(out/name).touch(exist_ok=False)
        for name in lock['base_ids']:
            bank,fixture=banks[name],fmap[name];policy=dict(lock['policy'],**fixture['policy_overrides'])
            analytic=name in FIX.ANALYTIC_BASE_IDS
            if not analytic and name not in FIX.OLD_BASE_IDS:raise ValueError('undeclared reference route')
            orders=[None] if analytic else lock['reference_policy']['orders'];values={};evidence={}
            zero=structural_branch(bank['input__mu'],bank['input__factor'])[1]
            for order in orders:
                tick=time.perf_counter();before=budget.used;result=None;attempted=False;available={};callmeta={};check=dict(passed=False)
                try:
                    if time.perf_counter()-phase>=lock['reference_policy']['max_phase_wall_seconds'] or (not analytic and budget.used>=budget.maximum):
                        raise RuntimeError('fixed global reference phase budget exhausted; no retry')
                    attempted=True
                    append(out/'reference_starts.jsonl',dict(id=name,order=order,route='analytic' if analytic else 'r27_numerical',started_unix=time.time(),budget_used_before=before,**SCOPE))
                    inputs={n:bank['input__'+n] for n in ('mu','factor','bounds','content_bias')}
                    if analytic:
                        result=REF.integrate_analytic(**inputs,policy=policy,deadline=phase+lock['reference_policy']['max_phase_wall_seconds'])
                    else:
                        result=REF.integrate_old(**inputs,policy=policy,reference_policy=lock['reference_policy'],order=order,budget=budget,
                            deadline=phase+lock['reference_policy']['max_phase_wall_seconds'])
                    available={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                    callmeta={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
                    check=analytic_check(result,bank,policy,lock['gates']) if analytic else reference_check(result,
                        inputs['mu'],inputs['factor'],inputs['bounds'],inputs['content_bias'],
                        policy,lock['reference_policy'],lock['gates'],order)
                    values[order]=(result['log_raw_mass'],check['passed'])
                except Exception as error:
                    if result is None:
                        available=dict(getattr(error,'available_arrays',{}))
                        callmeta=dict(getattr(error,'reference_metadata',{}),work=getattr(error,'reference_work',dict(cell_node_evaluations=0)))
                    check=dict(passed=False,failure_reason=type(error).__name__+': '+str(error))
                callmeta.update(check)
                consumed=callmeta['work']['cell_node_evaluations'];cumulative+=consumed
                used_cdf=callmeta['work'].get('analytic_cdf_evaluations',0);cdf_count+=used_cdf
                if budget.used-before!=consumed or budget.used!=cumulative:raise ValueError('partial/global reference node accounting differs')
                suffix='analytic' if analytic else str(order)
                for key,value in available.items():evidence[key+'__'+suffix]=value
                r=dict(callmeta,id=name,order=order,reference_kind='analytic' if analytic else 'r27_numerical',attempted=attempted,
                    analytic_cdf_evaluations=used_cdf,cumulative_analytic_cdf_evaluations=cdf_count,cumulative_cell_node_evaluations=cumulative,
                    phase_elapsed_seconds=time.perf_counter()-phase,wall_seconds=time.perf_counter()-tick,
                    available_array_sha256={k:array_sha(v) for k,v in available.items()},**SCOPE)
                append(out/'reference_results.jsonl',r);references.append(r);write(out/'reference_records.json',references)
                np.savez_compressed(directory/(name+'.npz'),**evidence)
            path=directory/(name+'.npz')
            comparison=dict(id=name,passed=False,reference_kind='analytic' if analytic else 'r27_numerical',
                archive_path=str(path.relative_to(out)),archive_sha256=sha(path),array_sha256={n:array_sha(v) for n,v in evidence.items()})
            if set(values)==set(orders):
                try:
                    args=(bank['input__bounds'],bank['input__content_bias'],policy['floor_density'],lock['gates'])
                    high=None if analytic else orders[-1]
                    if not analytic:comparison['reference']=compare_log_raw(values[orders[0]][0],values[high][0],*args,reference_comparison=True,structural_zero=zero)
                    if pmap[name]['passed']:
                        comparison['production']=compare_log_raw(bank['log_raw_mass'],values[high][0],*args,archived_probability=bank['probability'],structural_zero=zero)
                    comparison['passed']=bool(all(v[1] for v in values.values()) and (analytic or comparison['reference']['passed']) and comparison.get('production',{}).get('passed',False))
                except Exception as error:comparison['failure_reason']=type(error).__name__+': '+str(error)
            comparison.update(SCOPE);comparisons.append(comparison);write(out/'comparisons.json',comparisons)
            print('INDEPENDENT_REFERENCE',name,comparison['passed'],flush=True)
        for name in lock['identity_ids']:
            fixture=fmap[name];parent=fixture['identity']['parent']
            try:
                if not pmap[name]['passed'] or not pmap[parent]['passed']:raise ValueError('identity/base production failed')
                result=identity_check(fixture,banks[name],banks[parent],lock['gates'])
            except Exception as error:result=dict(id=name,parent=parent,passed=False,failure_reason=type(error).__name__+': '+str(error))
            result.update(SCOPE);identities.append(result);write(out/'identities.json',identities)
        limit=near_constant_check(banks,pmap,lock['gates']);limit.update(SCOPE);write(out/'near_constant_limit.json',limit)
        verify_joint_lock(run,out)
        elapsed,finished=time.perf_counter()-phase,time.time()
        resources=bool(elapsed<=lock['reference_policy']['max_phase_wall_seconds'] and cumulative<=budget.maximum)
        numerical=[r for r in references if r['reference_kind']=='r27_numerical'];analytic=[r for r in references if r['reference_kind']=='analytic']
        passed=bool(seal['all_production_calls_completed'] and seal['phase_wall_passed'] and resources and len(numerical)==40 and len(analytic)==3
            and len(comparisons)==23 and len(identities)==6 and limit['passed'] and all(r['passed'] for r in production+references+comparisons+identities))
        summary=dict(numerical_preflight_passed=passed,base_case_count=23,old_numerical_base_count=20,analytic_base_count=3,identity_count=6,
            reference_call_count=len(references),numerical_reference_call_count=len(numerical),analytic_reference_call_count=len(analytic),
            actual_reference_calls=sum(r['attempted'] for r in references),reference_calls_not_attempted=sum(not r['attempted'] for r in references),
            reference_cell_node_evaluations=cumulative,analytic_cdf_evaluations=cdf_count,reference_phase_wall_seconds=elapsed,reference_budget_passed=resources,
            production_phase_wall_seconds=seal['phase_wall_seconds'],production_phase_wall_passed=seal['phase_wall_passed'],
            production_failure_count=sum(not r['passed'] for r in production),reference_call_failure_count=sum(not r['passed'] for r in references),
            base_comparison_failure_count=sum(not r['passed'] for r in comparisons),identity_failure_count=sum(not r['passed'] for r in identities),near_constant_limit_passed=limit['passed'],
            started_unix=started,phase_finished_unix=finished,finished_unix=time.time(),
            next_action='fixed_synthetic_numerics_only_passed_separate_protocol_required' if passed else 'stop_fixed_implementation_preserve_all_failures',
            note='Old40 calls retain R27 limits; new3 are single restricted analytic CDF calls, not order refinement. Local LSE loss and cutoff bounds are not total floating-point certificates.',**SCOPE)
        write(out/'summary.json',summary)
        write(out/'verification.json',dict(delivery_integrity_passed=True,numerical_preflight_passed=passed,
            producer_lock_sha256=sha(run/'lock.json'),audit_lock_sha256=sha(out/'lock.json'),producer_completion_sha256=sha(run/'completion.json'),
            producer_forward_complete_sha256=sha(run/'forward_complete.json'),summary_sha256=sha(out/'summary.json'),source_count=len(lock['source_hashes']),
            producer_case_count=29,independent_reference_case_count=23,reference_call_count=43,numerical_reference_call_count=40,analytic_reference_call_count=3,
            identity_count=6,finished_unix=time.time(),**SCOPE))
        write(out/'completion.json',dict(files=inventory(out),verification_sha256=sha(out/'verification.json'),delivery_integrity_passed=True,numerical_preflight_passed=passed,
            finished_unix=time.time(),post_phase_bookkeeping_seconds_before_completion_write=time.perf_counter()-phase-elapsed,
            total_seconds_before_completion_write=time.perf_counter()-phase,**SCOPE))
    except (Exception,KeyboardInterrupt) as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,message=str(error),started_unix=started,finished_unix=time.time(),**SCOPE))
        write(out/'completion.json',dict(files={k:v for k,v in inventory(out).items() if k!='completion.json'},delivery_integrity_passed=False,
            numerical_preflight_passed=False,finished_unix=time.time(),**SCOPE))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','audit'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();{'freeze':freeze,'audit':audit}[args.command](args.run.resolve(),args.out.resolve())


if __name__=='__main__':main()
