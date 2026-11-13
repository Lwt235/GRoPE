"""Completion/identity/bounds checks for fixed formal work, without new model calls."""
import math
from experiments import native_task_writer_data as d
from experiments.native_task_writer_data import torch,np
from experiments.gaussian_scene_config import schedule

def main():
    torch.set_num_threads(4);records=[];pairing={};peak=0.
    assert d.sha(d.SCIENCE)==d.SCIENCE_SHA
    for seed in d.SEEDS:
        first=None
        for arm in d.ARMS:
            folder=d.OUT/'fits'/f'{arm}_{seed}';r=d.read(folder/'complete.json');assert r['complete'] and r['steps']==2000 and d.final(arm,seed).exists() and not d.latest(arm,seed).exists();assert d.sha(d.final(arm,seed))==r['sha256']
            ident=d.read(folder/'identity.json');common={k:ident[k] for k in ('module_sha256','parameters','native_parameters','schedule_sha256')}
            if first is None:first=common
            else:assert common==first,(seed,arm)
            logs=[d.json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()];assert [x['step'] for x in logs]==list(range(1,2001));order=schedule(d.plan(),seed)
            assert [list(k) for row in logs for k in row['samples']]==[list(k) for k in order[:4000]]
            assert all(np.isfinite(row['loss']) and np.isfinite(row['gradient_norm']) for row in logs);perfield=[]
            for meta in d.metas():
                rid=meta['id'];f=d.load(d.fieldpath(arm,seed,rid));base=d.load(d.old.scene_path(rid));m=d.read(d.OUT/'formation'/f'{arm}_{seed}_{rid}.json');e=d.read(d.OUT/'evaluation'/f'{arm}_{seed}_{rid}.json');assert m['sha256']==e['field_sha256']==d.sha(d.fieldpath(arm,seed,rid));assert d.sha(d.old.scene_path(rid))==m['source_input_sha256']
                assert e['complete'] and e['predictions']==16 and e['field_built_once'] and e['field_loaded_once'] and e['query_writer_calls']==0
                assert f['features'].dtype==torch.float16 and f['means'].dtype==torch.float32 and f['scales'].dtype==torch.float32 and f['rotations'].dtype==torch.float32
                assert all(torch.isfinite(x).all() for x in f.values() if torch.is_tensor(x));assert (f['scales']>0).all()
                assert all(torch.equal(f[k],base[k]) for k in ('has_content','source_ids','unit','source_c2w','source_intrinsic','mass','colors'))
                if arm[0]=='C':assert all(torch.equal(f[k],base[k]) for k in ('means','scales','rotations'))
                R=base['source_c2w'][0,:3,:3];center=(f['means']-base['means'])@R/base['unit'];logscl=(f['scales']/base['scales']).log();qb=torch.nn.functional.normalize(base['rotations'],dim=-1);qf=torch.nn.functional.normalize(f['rotations'],dim=-1);angle=2*torch.acos((qb*qf).sum(-1).abs().clamp(max=1))
                assert center.abs().max()<=.250001 and logscl.abs().max()<=.700001 and angle.max()<=math.pi/6+.00001
                assert torch.equal(f['features'][~base['has_content']],base['features'][~base['has_content']]);assert torch.equal(f['opacities'][~base['has_content']],base['opacities'][~base['has_content']])
                perfield.append(dict(id=rid,center_normalized_max_abs=float(center.abs().max()),log_scale_max_abs=float(logscl.abs().max()),rotation_max_rad=float(angle.max()),unit=float(f['unit']),valid_slots=int(f['has_content'].sum()),features_shape=list(f['features'].shape),state_bytes=d.fieldpath(arm,seed,rid).stat().st_size))
            records.append(dict(arm=arm,seed=seed,updates=2000,final_sha256=r['sha256'],fields=perfield));pairing[str(seed)]=first
    d.write(d.OUT/'completion_checks.json',dict(complete=True,fits=18,retained_updates=36000,formal_training_samples=72000,evaluation_predictions=12672,permanent_source_only_states=792,all_exact2000_and_original_sampling=True,paired_initial_and_schedule_identity=pairing,records=records,learned_geometry_accuracy_claim=False,all_data_used_development=True))
    print(dict(completion_checks=True,fits=18,fields=792),flush=True)
if __name__=='__main__':main()
