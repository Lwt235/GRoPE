"""Standard-library synthetic reporting checks; no experiment/model imports."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import round17_size_temperature_report_extract as R


def inventory(root):
    return {str(p.relative_to(root)):R.digest(p) for p in root.rglob('*') if p.is_file() and p != root/'completion.json'}


def fixture(base):
    run, audit, repo = base/'run', base/'audit', base/'repo'
    for path in (run, audit, repo):path.mkdir()
    sources = {}
    for i in range(84):
        name=f'source_{i}.py';(repo/name).write_text('# synthetic source\n')
        sources[name]=R.digest(repo/name)
        for directory in (run, audit) if i<37 else (run,):
            (directory/'source_snapshot').mkdir(exist_ok=True)
            (directory/'source_snapshot'/name).write_bytes((repo/name).read_bytes())
    entries=[dict(scene=s,pair=[i,i+1],path=f'/synthetic/{s}/{i:02}') for s in R.SCENES for i in range(16)]
    lock=dict(created_unix=1.,source_hashes=sources,roles=list(R.ROLES),config=R.CONFIG,entries=entries,
              prior=dict(entries=entries,files={}),runtime=dict(cpu_threads=4))
    R.write(run/'lock.json',lock);(run/'lock.sha256').write_text(R.digest(run/'lock.json'))
    design=dict(created_unix=2.,run=str(run),producer_lock_sha256=R.digest(run/'lock.json'),
                source_hashes=dict(list(sources.items())[:37]),**R.AUDIT_POLICY)
    R.write(audit/'lock.json',design);(audit/'lock.sha256').write_text(R.digest(audit/'lock.json'))
    chain={}
    for s in R.SCENES:
        directory=run/'forward'/s.replace('/','__');directory.mkdir(parents=True)
        R.write(directory/'started.json',dict(created_unix=3.,lock_sha256=R.digest(run/'lock.json'),runtime=lock['runtime']))
        # Binary evidence is hashed, never decoded by the report extractor.
        (directory/'synthetic.npz').write_bytes(b'not an array archive')
        R.write(directory/'completion.json',dict(stage='forward',scene=s,lock_sha256=R.digest(run/'lock.json'),
            record_count=448,runtime=lock['runtime'],started_unix=3.,finished_unix=4.,labels_decoded=False,
            primary_exact_matrices=224,determinism_matrices=28,all_probabilities_retained=True,files=inventory(directory)))
        chain[s]=dict(completion_sha256=R.digest(directory/'completion.json'),finished_unix=4.)
    R.write(run/'forward_complete.json',dict(lock_sha256=R.digest(run/'lock.json'),labels_decoded=False,
        all_probabilities_retained=True,matrices=1344,raw_probability_bytes=512434944,sealed_unix=5.,forward_chain=chain))
    for s in R.SCENES:
        directory=run/'evaluation'/s.replace('/','__');directory.mkdir(parents=True)
        R.write(directory/'started.json',dict(created_unix=6.,forward_complete_sha256=R.digest(run/'forward_complete.json')))
        rows=[];paired=[];strata=[]
        for entry in (e for e in entries if e['scene']==s):
            for c in R.CONDITIONS:
                q,k=c.split('_')
                for role in R.ROLES:
                    rows.append(dict(scene=s,pair=entry['pair'],condition=c,role=role,config=R.CONFIG,
                                     shape=[R.NATIVE_COUNTS[q],R.NATIVE_COUNTS[k]],dtype='float64'))
                paired.append(dict(scene=s,pair=entry['pair'],condition=c))
                strata.extend(dict(scene=s,pair=entry['pair'],condition=c,width=w,height=h) for w,h in R.SIZES[q])
        for name,values in (('rows',rows),('paired_rows',paired),('size_strata',strata)):
            R.write(directory/(name+'.json'),values)
        R.write(directory/'completion.json',dict(stage='evaluation',scene=s,lock_sha256=R.digest(run/'lock.json'),
            record_count=448,runtime=lock['runtime'],started_unix=6.,finished_unix=7.,labels_decoded=True,
            forward_complete_sha256=R.digest(run/'forward_complete.json'),files=inventory(directory)))
    value=dict(kl=.3,js=.1,direct_kl=.4,direct_js=.2,mass=100,n_point=16,pck_count=8,pck=.5)
    metrics={role:dict(by_scene_condition={s:{c:dict(value) for c in R.CONDITIONS} for s in R.SCENES},
        by_condition={c:dict(pck=.5,pck_exact=dict(numerator=1,denominator=2)) for c in R.CONDITIONS},
        common9_kl=.3,direct5_kl=.4,J_descriptive=.35) for role in R.ROLES}
    def paired_groups(names):
        return {name:dict(by_scene={s:dict(pck_difference_exact=dict(numerator=-1,denominator=16)) for s in R.SCENES},
                          macro=dict(pck_difference_exact=dict(numerator=-1,denominator=16))) for name in names}
    groups=[]
    for s in R.SCENES:
        for q in R.PARTITIONS:
            for i,(w,h) in enumerate(R.SIZES[q]):
                count=R.NATIVE_COUNTS[q] if q!='mixed' else (81 if i==0 else 54)
                groups.append(dict(scene=s,query=q,width=w,height=h,n=16*count,tau_obs32=dict(median=1.)))
    summary=dict(created_unix=8.,lock_sha256=R.digest(run/'lock.json'),forward_complete_sha256=R.digest(run/'forward_complete.json'),
        config=R.CONFIG,roles=list(R.ROLES),metrics=metrics,paired_summary=paired_groups(R.CONDITIONS),
        size_strata_summary=paired_groups(sorted(R.strata_names())),support_statistics=dict(unique_queries=39888,groups=groups),
        matrices=1344,pairs=48,paired_conditions=672,size_stratum_rows=1104,descriptive_attribution_only=True,
        independent_audit_pending=True,**{n:False for n in R.SUMMARY_FALSE})
    R.write(run/'summary.json',summary)
    R.write(run/'completion.json',dict(finished_unix=9.,lock_sha256=R.digest(run/'lock.json'),
        summary_sha256=R.digest(run/'summary.json'),independent_audit_pending=True,files=inventory(run)))
    R.write(audit/'started.json',dict(created_unix=10.,producer_lock_sha256=R.digest(run/'lock.json')))
    numeric=[]
    for s in R.SCENES:
        for c in R.CONDITIONS:
            q,k=c.split('_')
            for role in R.ROLES:
                numeric.append(dict(scene=s,pair=[0,1],condition=c,query=q,key=k,role=role,
                    indices=[i*(R.NATIVE_COUNTS[q]-1)//7 for i in range(8)],passed=True,
                    same_prepared_tau32_exact=True,same_prepared_bias32_exact=True,max_probability_error=0.,
                    reference_convergence_error=0.,same_mass_probability_error=0.,hybrid_probability_error=0.))
    R.write(audit/'numerical_checks.json',numeric)
    R.write(audit/'independent_summary.json',{n:summary[n] for n in ('metrics','paired_summary','size_strata_summary','support_statistics')})
    verification=dict(passed=True,descriptive_attribution_only=True,**{n:False for n in R.VERIFICATION_FALSE},
        **{'producer_'+n+'_sha256':R.digest(run/(n+'.json')) for n in ('lock','forward_complete','summary','completion')},
        independent_audit_lock_sha256=R.digest(audit/'lock.json'),original_design_created_unix=2.,started_unix=10.,finished_unix=11.,
        matrices=1344,primary_old_hash_matches=672,independent_count_tables=672,independent_numerical_matrices=84,
        numerical_query_rows=672,actual_prepared_bias_sample_matrices=84,unique_support_queries=39888,
        numerical_failure_count=0,maximum_metric_sum_error=0.,maximum_summary_error=0.,maximum_complete_same_raw_error=0.)
    R.write(audit/'verification.json',verification)
    R.write(audit/'completion.json',dict(passed=True,scientific_gate_passed=False,finished_unix=12.,files=inventory(audit)))
    return run,audit,repo


class ReportTests(unittest.TestCase):
    def test_missing_audit_reads_no_effects(self):
        with TemporaryDirectory() as folder, patch.object(R,'read',side_effect=AssertionError('premature read')):
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'completed independent acceptance'):
                R.accepted_metadata(root/'run',root/'audit')

    def test_full_metadata_chain_and_all_tables_preserve_fractions_and_scope(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);run,audit,repo=fixture(root)
            receipt=R.extract(run,audit,root/'report',repo=repo)
            self.assertEqual(receipt['table_rows'],dict(role_scene_conditions=84,role_condition_macros=28,role_overall_macros=2,
                paired_scene_and_macro=56,size_strata_scene_and_macro=92,unique_support_groups=21,
                all_evaluated_rows=1344,all_paired_rows=672,all_size_stratum_rows=1104))
            self.assertFalse(receipt['scientific_gate_passed'])
            self.assertFalse(receipt['general_research_goal_achieved'])
            rows=R.read(root/'report/paired_scene_and_macro.json')
            self.assertTrue(all(Fraction(**r['pck_difference_exact'])==Fraction(-1,16) for r in rows))
            self.assertTrue(R.read(run/'summary.json')['independent_audit_pending'])
            for name,sha in receipt['outputs_sha256'].items():self.assertEqual(R.digest(root/'report'/name),sha)

    def test_cross_run_verification_rejected_after_valid_outer_rehash(self):
        for field in ('lock','forward_complete','summary','completion'):
            with self.subTest(field=field), TemporaryDirectory() as folder:
                root=Path(folder);run,audit,repo=fixture(root)
                value=R.read(audit/'verification.json');value['producer_'+field+'_sha256']='0'*64
                (audit/'verification.json').unlink();R.write(audit/'verification.json',value)
                complete=R.read(audit/'completion.json');complete['files']=inventory(audit)
                (audit/'completion.json').unlink();R.write(audit/'completion.json',complete)
                with self.assertRaisesRegex(ValueError,'different producer run'):
                    R.accepted_metadata(run,audit,repo=repo)

    def test_failed_audit_precedes_summary_access(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);run,audit,repo=fixture(root)
            complete=R.read(audit/'completion.json');complete['passed']=False
            (audit/'completion.json').unlink();R.write(audit/'completion.json',complete)
            original=R.read
            def guarded(path):
                self.assertNotEqual(Path(path).name,'summary.json')
                return original(path)
            with patch.object(R,'read',side_effect=guarded), self.assertRaisesRegex(ValueError,'successful descriptive audit'):
                R.accepted_metadata(run,audit,repo=repo)

    def test_inventory_missing_extra_changed_and_unsafe_paths(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'x.bin').write_bytes(b'array bytes');files=inventory(root)
            R.inventory(root,files)
            (root/'added').write_text('x')
            with self.assertRaisesRegex(ValueError,'population'):R.inventory(root,files)
            (root/'added').unlink();(root/'x.bin').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):R.inventory(root,files)
            with self.assertRaisesRegex(ValueError,'unsafe'):R.inventory(root,{'../x':'0'})

    def test_output_inside_either_frozen_root_stops_before_reading(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);run,audit=root/'run',root/'audit'
            for out in (root,run,run/'report',audit,audit/'report'):
                with patch.object(R,'accepted_metadata',side_effect=AssertionError('premature acceptance')):
                    with self.assertRaisesRegex(ValueError,'outside both frozen roots'):R.extract(run,audit,out)

    def test_population_and_fraction_mutations_rejected(self):
        with TemporaryDirectory() as folder:
            run,audit,repo=fixture(Path(folder));data=R.accepted_metadata(run,audit,repo=repo)
            for name in ('evaluated','paired','strata'):
                changed=deepcopy(data);changed[name]=changed[name][:-1]
                with self.assertRaises(ValueError):
                    R.verify_population(changed['summary'],changed['evaluated'],changed['paired'],changed['strata'],changed['lock'])
            for bad in (dict(numerator=2,denominator=4),dict(numerator=True,denominator=1)):
                with self.assertRaises(ValueError):R.check_fraction(bad)


if __name__=='__main__':unittest.main()
