"""Two complete paired seeds, all old development places; descriptive means only."""
import numpy as np
from experiments.source_geometry_data import *

def main():
    p=plan();rows=read(OUT/'partial_scoring/metrics.json')['rows'];seeds=list(SEEDS[:2]);kinds=['P_pred','G_pred','P_sensor','G_sensor','parent_L']
    endpoints=[('missing','roi','centered_log_mse'),('missing','roi','log_mse'),('full','roi','centered_log_mse'),('full','whole','absrel')]
    summaries={};comparisons={};interactions={}
    for condition,region,metric in endpoints:
        key='/'.join([condition,region,metric]);arrays={}
        for k in kinds:
            array=np.zeros((2,12))
            for si,seed in enumerate(seeds):
                for pi,meta in enumerate(p['catalog']):
                    group=[r['metrics'][region][metric] for r in rows if r['seed']==seed and r['kind']==k and r['place']==meta['place'] and r['condition']==condition]
                    assert len(group)==4
                    array[si,pi]=np.mean(group)
            arrays[k]=array;summaries[key+'/'+k]=dict(mean=float(array.mean()),per_seed=array.mean(1).tolist(),per_place=array.mean(0).tolist())
        for a,b in [('G_pred','P_pred'),('G_sensor','P_sensor'),('P_sensor','P_pred'),('G_sensor','G_pred')]+[(k,'parent_L') for k in kinds[:4]]:
            x,y=arrays[a],arrays[b];comparisons[key+'/'+a+'-'+b]=dict(relative_error_reduction_percent=float(100*(y.mean()-x.mean())/y.mean()),per_seed_relative_percent=(100*(y.mean(1)-x.mean(1))/y.mean(1)).tolist(),improved_places=int((x.mean(0)<y.mean(0)).sum()))
        delta=(arrays['P_sensor']-arrays['G_sensor'])-(arrays['P_pred']-arrays['G_pred']);interactions[key]=dict(absolute=float(delta.mean()),per_seed_absolute=delta.mean(1).tolist(),per_place_absolute=delta.mean(0).tolist())
    result=dict(complete_paired_seeds=2,seeds=seeds,all12places_retained=True,summaries=summaries,comparisons=comparisons,interactions=interactions,scope='Two-seed descriptive means only; third seed and final paired intervals remain required. No change of training configuration or method decision, no fresh confirmation.')
    write_json(OUT/'two_seed_four_arm_description.json',result)
    endpoint='missing/roi/centered_log_mse'
    for k in kinds:print(k,summaries[endpoint+'/'+k])
    for pair in ['G_pred-P_pred','G_sensor-P_sensor','G_sensor-parent_L']:print(pair,comparisons[endpoint+'/'+pair])
    print('interaction',interactions[endpoint])

if __name__=='__main__':main()
