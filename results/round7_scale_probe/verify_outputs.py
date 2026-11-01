"""Independent output arithmetic/invariants; no imports of the probe's analysis."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

directory=Path(__file__).resolve().parent
protocol=json.loads((directory/'frozen_protocol.json').read_text())
summary=json.loads((directory/'summary.json').read_text())
labels=json.loads((directory/'label_diagnostics.json').read_text())
roles={role:json.loads((directory/f'{role}_rows.json').read_text())['rows'] for role in protocol['roles']}
grids=protocol['grids']
checks=[];errors=[]
def check(name,value):
    checks.append({'name':name,'pass':bool(value)})
    if not value:errors.append(name)
def ident(r):return r['scene'],r['grid'],r['q_index'],r['k_index']
label_byid={ident(r):r for r in labels}
check('3040 unique pair/grid label records',len(labels)==len(label_byid)==3040)
expected_ids=set(label_byid)
for role,rows in roles.items():
    check(role+' exact identities',len(rows)==len(expected_ids) and {ident(r) for r in rows}==expected_ids)
    check(role+' same visible/point masks',all(abs(r['visible_mass']-label_byid[ident(r)]['visible_mass'])<1e-8 and
        r['n_valid']==label_byid[ident(r)]['n_valid'] and r['n_point']==label_byid[ident(r)]['oracle']['n_point'] for r in rows))
    check(role+' positive and finite metrics',all(np.isfinite([r['ce_sum'],r['kl_sum'],r['js_sum']]).all() and
        r['kl_sum']>=-1e-5 and r['js_sum']>=-1e-5 for r in rows))
    check(role+' point predictions obey grid ceilings',all(r['pck8_count']<=label_byid[ident(r)]['oracle']['pck8_count'] and
        r['pck_stride_count']<=label_byid[ident(r)]['oracle']['pck_stride_count'] for r in rows))
pairmass=defaultdict(list)
for r in labels:
    pairmass[r['scene'],r['q_index'],r['k_index']].append(r['visible_mass'])
check('every source image has27648 samples',all(r['sample_count']==192*144 and sum(r['counts_by_reason'])==192*144 and
    r['counts_by_reason'][0]==r['visible_mass'] for r in labels))
check('sensor area visibility identical for all five grids',len(pairmass)==608 and all(len(v)==5 and max(v)==min(v) for v in pairmass.values()))
max_error=0.;ci_error=0.
fields={'kl':('kl_sum','visible_mass'),'js':('js_sum','visible_mass'),
    'pck8':('pck8_count','n_point'),'pck_stride':('pck_stride_count','n_point')}
for split,n in [('val',2),('test',5)]:
    scenes=sorted({r['scene'] for r in labels if r['split']==split})
    check(split+' expected scene count',len(scenes)==n)
    metric_values={}
    for role,rows in roles.items():
        for scene in scenes:
            for grid in grids:
                chosen=[r for r in rows if r['scene']==scene and r['grid']==grid]
                check(role+':'+scene+':'+grid+' complete pair count',len(chosen)==(96 if split=='test' else 64))
                metric_values[role,scene,grid]={name:sum(r[numerator] for r in chosen)/sum(r[denominator] for r in chosen)
                    for name,(numerator,denominator) in fields.items()}
        for name in fields:
            computed=np.mean([metric_values[role,s,g][name] for s in scenes for g in grids])
            max_error=max(max_error,abs(computed-summary[split]['roles'][role]['macro'][name]))
    for grid in grids+['all_new_grids']:
        subset=grids if grid=='all_new_grids' else [grid]
        a=np.array([np.mean([metric_values['full',s,g]['kl'] for g in subset]) for s in scenes])
        b=np.array([np.mean([metric_values['fixed',s,g]['kl'] for g in subset]) for s in scenes])
        reported=summary[split]['full_minus_fixed_kl'][grid]
        max_error=max(max_error,abs(a.mean()/b.mean()-1-reported['relative']))
        check(split+':'+grid+' place improvement arithmetic',int((a<b).sum())==reported['improving_places'])
        if split=='test':
            rng=np.random.default_rng(20261030);draw=rng.integers(0,n,(20000,n))
            rel=np.array([a[index].mean()/b[index].mean()-1 for index in draw])
            ci_error=max(ci_error,float(np.max(abs(np.quantile(rel,[.025,.975])-reported['descriptive_relative_ci95']))))
check('independent macro arithmetic exact to1e-12',max_error<1e-12)
check('independent paired place CI exact to1e-12',ci_error<1e-12)
result=dict(checks=checks,failures=errors,max_macro_arithmetic_error=max_error,max_ci_error=ci_error,
    roles=len(roles),rows=sum(map(len,roles.values())),label_conditions=len(labels),
    scope='output arithmetic, common mask/area invariants and point quantization ceilings; no extra model prediction replay',
    files_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.glob('*.json') if path.name!='independent_checks.json'})
(directory/'independent_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','files_sha256')},indent=2))
if errors:raise SystemExit(1)
