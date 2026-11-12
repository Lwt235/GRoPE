"""Descriptive complete-seed summaries only; never a fixed-method decision or fresh confirmation."""
import numpy as np
from experiments.source_geometry_data import *

def main():
    p=plan();rows=read(OUT/'partial_scoring/metrics.json')['rows'];seed=SEEDS[0]
    kinds=['P_pred','G_pred','P_sensor','G_sensor','parent_L'];endpoints=[('missing','roi','centered_log_mse'),('missing','roi','log_mse'),('full','roi','centered_log_mse'),('full','whole','absrel')]
    summaries={}
    for c,region,metric in endpoints:
        for k in kinds:
            by_place={}
            for meta in p['catalog']:
                group=[r['metrics'][region][metric] for r in rows if r['seed']==seed and r['kind']==k and r['place']==meta['place'] and r['condition']==c]
                assert len(group)==4
                by_place[meta['place']]=float(np.mean(group))
            summaries[f'{c}/{region}/{metric}/{k}']=dict(mean=float(np.mean(list(by_place.values()))),per_place=by_place)
    effects={}
    for endpoint in ['/'.join(x) for x in endpoints]:
        for a,b in [('G_pred','P_pred'),('G_sensor','P_sensor'),('P_sensor','P_pred'),('G_sensor','G_pred')]+[(k,'parent_L') for k in kinds[:4]]:
            x=summaries[endpoint+'/'+a];y=summaries[endpoint+'/'+b]
            effects[endpoint+'/'+a+'-'+b]=dict(relative_error_reduction_percent=100*(y['mean']-x['mean'])/y['mean'],absolute_error_reduction=y['mean']-x['mean'],improved_places=sum(x['per_place'][m]<y['per_place'][m] for m in x['per_place']))
    endpoint='missing/roi/centered_log_mse';dp=effects[endpoint+'/G_pred-P_pred']['absolute_error_reduction'];ds=effects[endpoint+'/G_sensor-P_sensor']['absolute_error_reduction']
    result=dict(complete_paired_seeds=1,seed=seed,all12places_retained=True,summaries=summaries,effects=effects,primary_interaction_absolute=ds-dp,
                scope='One complete paired seed descriptive means only; remaining2seeds unchanged. No uncertainty/fresh-confirmation claim or method promotion.')
    write_json(OUT/'first_seed_four_arm_description.json',result)
    for kind in kinds:print(kind,summaries[endpoint+'/'+kind]['mean'])
    for key in [endpoint+'/G_pred-P_pred',endpoint+'/G_sensor-P_sensor']:print(key,effects[key])
    print('primary interaction',ds-dp,flush=True)

if __name__=='__main__':main()
