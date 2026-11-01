"""Read-only complete reanalysis of the sealed R15 Stage A population."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round15_feasibility as A
from experiments import round16_mechanism_metrics as M

ROOT = A.ROOT
UPSTREAM = ROOT/'results/round15_feasibility'
OWN_SOURCES = (
    'ROUND16_MECHANISM_PROTOCOL.md', 'docs/round15_pre_result_mechanism_hypotheses.md',
    'experiments/round16_mechanism.py', 'tests/test_round16_mechanism.py',
    'experiments/round16_mechanism_metrics.py', 'tests/test_round16_mechanism_metrics.py',
    'experiments/round16_mechanism_audit_metrics.py', 'tests/test_round16_mechanism_audit_metrics.py',
    'experiments/round16_mechanism_audit.py', 'tests/test_round16_mechanism_audit.py',
)
sha, read, write, array_sha = A.sha, A.read, A.write, A.array_sha


def absolute(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def sources():
    return dict(A.sources(), **{p:sha(ROOT/p) for p in OWN_SOURCES})


def upstream_files():
    lock=A.verify_lock(UPSTREAM)
    complete=read(UPSTREAM/'completion.json')
    files={str(UPSTREAM/p):h for p,h in complete['files'].items()}
    for name in ('completion.json','decision.json'):
        files[str(UPSTREAM/name)]=sha(UPSTREAM/name)
    decision_path=ROOT/'results/round15_regional_post_result_decision.json'
    decision=read(decision_path)
    if (decision['action']!='stop_fixed_quartic_candidate_before_old_validation'
            or decision['scientific_gate_passed'] or decision['general_research_goal_achieved']
            or decision['old_validation']['source_freeze_performed']):
        raise ValueError('complete retained R15 futility decision required')
    files[str(decision_path)]=sha(decision_path)
    for v in (*decision['bindings'].values(),decision['report']):files[v['path']]=v['sha256']
    files.update(lock['inputs']);files.update(lock['manifests'])
    for e in lock['target_contract']['entries']:files[e['target_path']]=e['target_sha256']
    A.verify_files({str(absolute(p)):h for p,h in files.items()})
    return lock,{str(absolute(p)):h for p,h in files.items()}


def freeze(out):
    if out.exists():raise ValueError('refusing diagnostic freeze overwrite')
    prior,files=upstream_files()
    hashes=sources();out.mkdir(parents=True)
    for p in hashes:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/p).read_bytes())
    value=dict(created_unix=time.time(),source_hashes=hashes,upstream_files=files,
        stage_a_root=str(UPSTREAM),entries=prior['target_contract']['entries'],
        expected_rows=5184,expected_paired_rows=3456,expected_query_states=1016064,
        expected_static_queries=28224,fields=list(A.FIELDS),runtime=A.configure_runtime('cpu'),
        joint_producer_and_independent_audit_freeze=True,diagnostic_only=True,
        new_predictions=False,new_data=False,scientific_gate_passed=False)
    write(out/'lock.json',value);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN_MECHANISM',sha(out/'lock.json'),flush=True)


def verify_lock(out):
    lock=read(out/'lock.json')
    if (sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip()
            or lock['source_hashes']!=sources() or lock['runtime']!=A.configure_runtime('cpu')):
        raise ValueError('changed diagnostic source/runtime/lock')
    A.verify_files({str(out/'source_snapshot'/p):h for p,h in lock['source_hashes'].items()})
    A.verify_files(lock['upstream_files'])
    prior=read(UPSTREAM/'lock.json')
    if lock['entries']!=prior['target_contract']['entries']:raise ValueError('changed cohort')
    return lock


def reconstructed_probability(p0,content32,temperature):
    logits=np.log(p0)+(np.float32(temperature)*content32).astype(np.float64)
    shifted=np.exp(logits-logits.max(axis=1,keepdims=True))
    return shifted/shifted.sum(axis=1,keepdims=True)


def identity(row):
    return row['scene'],tuple(row['pair']),row['family'],row['grid_index'],row['query']


def population(rows,entries,families):
    expected={(e['scene'],tuple(e['pair']),f,i,q) for e in entries for f in families
              for i in range(12) for q in A.QUERIES}
    ids=[identity(r) for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('diagnostic population mismatch')
    for row in rows:
        if row['config']!=A.GRID[row['grid_index']]:raise ValueError('configuration changed')


@torch.no_grad()
def extract(out,scene):
    lock=verify_lock(out);prior=read(UPSTREAM/'lock.json')
    entries=A.scene_entries(prior,scene);directory=out/'analysis'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing existing analysis')
    directory.mkdir(parents=True);started=time.time()
    write(directory/'started.json',dict(created_unix=started,lock_sha256=sha(out/'lock.json')))
    original_dir=UPSTREAM/'forward'/scene.replace('/','__')
    records={identity(r):r for r in read(original_dir/'records.json')}
    original_metrics={identity(r):r for r in read(UPSTREAM/'evaluation'/scene.replace('/','__')/'rows.json')}
    parts=A.S.read_parts(UPSTREAM);tparts=A.S.M.partition_tensors(parts,'cpu')
    key_px=parts['g8']['mean']*8
    rows=[];paired=[];static_count=0;state_count=0;maximum_reconstruction=0.
    for pair_index,entry in enumerate(entries):
        input_path=absolute(entry['path'])/'inputs.npz'
        inputs=A.load_forward_inputs(input_path)
        observations=A.S.A.observation_tensors(inputs,'cpu')
        key=A.S.M.prepare_key(observations,tparts['g8'])
        labels=A.load_center_targets(absolute(entry['target_path']))
        archive_path=original_dir/(Path(entry['path']).name+'.npz')
        archive_digest=sha(archive_path)
        with np.load(archive_path,allow_pickle=False) as archive:
            if set(archive.files)!=set(A.expected_arrays()):raise ValueError('original array population changed')
            for q in A.QUERIES:
                query=A.prepare_query(observations,tparts[q],True)
                mu=(query['mean']*8).numpy();cov=query['cov'].numpy()
                content=(query['features']@key['features'].T).numpy()
                target=labels[q+'_point_xy'];valid=labels[q+'_point_visible']
                saved=dict(mu_px=mu,cov_reference8=cov,key_px=key_px,content32=content,target_px=target,valid=valid)
                relative=str((directory/'queries'/(Path(entry['path']).name+'__'+q+'.npz')).relative_to(out))
                diagnostics={};static_count+=len(mu)
                for family in A.FAMILIES:
                    for i,cfg in enumerate(A.GRID):
                        rowid=(scene,tuple(entry['pair']),family,i,q);record=records[rowid]
                        p=archive[record['field']];base=records[(scene,tuple(entry['pair']),family,3*(i//3),q)]
                        p0=archive[base['field']]
                        if (array_sha(p)!=record['array_sha256'] or array_sha(p0)!=base['array_sha256']
                                or record['archive_sha256']!=archive_digest):
                            raise ValueError('original sealed probability changed')
                        reconstruct=reconstructed_probability(p0,content,cfg['temperature'])
                        error=float(np.max(np.abs(reconstruct-p)));maximum_reconstruction=max(maximum_reconstruction,error)
                        if error>1e-12:raise ValueError('temperature probability identity failed')
                        data=M.analyze(p0,p,content,mu,cov,key_px,target,valid,**cfg)
                        if float(np.max(data['identity_error']))>1e-10:raise ValueError('winner margin identity failed')
                        old=original_metrics[rowid]
                        pck=int((data['correct']==1).sum());winner_sha=array_sha(data['winner'])
                        if (pck!=old['pck_count'] or int(valid.sum())!=old['n_point']
                                or winner_sha!=old['argmax_sha256']):
                            raise ValueError('original PCK/argmax changed')
                        prefix=f'{family}__c{i:02d}__'
                        saved.update({prefix+k:v for k,v in data.items()});diagnostics[(family,i)]=data
                        rows.append(dict(scene=scene,pair=entry['pair'],family=family,grid_index=i,query=q,
                            config=cfg,stats=M.stats(data),diagnostic_path=relative,
                            diagnostic_array_hashes={k:array_sha(v) for k,v in data.items()},
                            probability_sha256=record['array_sha256'],p0_sha256=base['array_sha256'],
                            argmax_sha256=winner_sha,pck_count=pck,n_point=int(valid.sum()),n_query=len(mu),
                            input_sha256=record['input_sha256'],target_sha256=entry['target_sha256']))
                        state_count+=len(mu)
                for comparison in ('quartic_point','gaussian_point'):
                    for i,cfg in enumerate(A.GRID):
                        paired.append(dict(scene=scene,pair=entry['pair'],family='full_vs_'+comparison,
                            grid_index=i,query=q,config=cfg,
                            stats=M.paired_stats(diagnostics[('quartic_full',i)],diagnostics[(comparison,i)])))
                destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination,**saved)
        print('MECHANISM_EXTRACT',scene,pair_index+1,'/16',flush=True)
    population(rows,entries,A.FAMILIES);population(paired,entries,('full_vs_quartic_point','full_vs_gaussian_point'))
    write(directory/'rows.json',rows);write(directory/'paired_rows.json',paired)
    A.complete_stage(directory,out,'analysis',scene,started,row_count=len(rows),paired_row_count=len(paired),
        query_state_count=state_count,static_query_count=static_count,maximum_reconstruction_error=maximum_reconstruction,
        diagnostic_only=True,new_predictions=False)


def summarize(out):
    lock=verify_lock(out)
    if (out/'summary.json').exists():raise ValueError('refusing summary overwrite')
    rows=[];paired=[];states=static=0;reconstruction=0.
    for scene in A.SCENES:
        directory=out/'analysis'/scene.replace('/','__')
        complete=A.verify_stage(directory,out,'analysis',scene)
        rows.extend(read(directory/'rows.json'));paired.extend(read(directory/'paired_rows.json'))
        states+=complete['query_state_count'];static+=complete['static_query_count']
        reconstruction=max(reconstruction,complete['maximum_reconstruction_error'])
    population(rows,lock['entries'],A.FAMILIES)
    population(paired,lock['entries'],('full_vs_quartic_point','full_vs_gaussian_point'))
    if (len(rows),len(paired),states,static)!=(5184,3456,1016064,28224):raise ValueError('incomplete diagnostic counts')
    write(out/'summary.json',dict(metrics=M.summarize(rows),paired_metrics=M.summarize(paired),
        row_count=len(rows),paired_row_count=len(paired),query_state_count=states,static_query_count=static,
        maximum_reconstruction_error=reconstruction,created_unix=time.time(),diagnostic_only=True,
        new_predictions=False,scientific_gate_passed=False,independent_audit_pending=True))
    write(out/'completion.json',dict(lock_sha256=sha(out/'lock.json'),
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},finished_unix=time.time()))
    print('MECHANISM_COMPLETE_AUDIT_PENDING',len(rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','extract','summarize'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round16_mechanism')
    parser.add_argument('--scene',choices=A.SCENES);args=parser.parse_args()
    if args.stage=='extract':extract(args.out,args.scene)
    else:globals()[args.stage](args.out)


if __name__=='__main__':main()
