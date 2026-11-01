"""Post-run independent saved-transport metric and full-kernel NumPy audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.round8_oracle_analysis import (
    ROOT,sha,load_validation_sources,numpy_rows,numpy_macro,compare_tree,identity,
)


def main():
    directory=ROOT/'results/round8_transport_probe'
    result=json.loads((directory/'summary.json').read_text())
    lockpath=ROOT/'results/round8_oracle_development/lock.json'
    lock=json.loads(lockpath.read_text())
    assert sha(lockpath)==result['lock_sha256']
    assert result['source_hashes']==lock['source_hashes']
    assert result['source_paths']['val']==lock['data_sources']['val']
    assert result['source_paths']['oracle_val']==lock['data_sources']['oracle_val']
    for path,digest in result['source_hashes'].items():assert sha(ROOT/path)==digest
    assert sha(directory/'predictions.npz')==result['prediction_sha256']
    labels,ids,sources=load_validation_sources(result['source_paths']['val'],result['source_paths']['oracle_val'])
    masks=dict(all_query_valid=labels['valid'],oracle_eligible=labels['valid']&labels['oracle_covisible'],
               predicted_fallback=labels['valid']&~labels['oracle_covisible'])
    with np.load(directory/'predictions.npz',allow_pickle=False) as data:
        predictions={name:data[name].copy() for name in data.files}
    checks=[];aggregates={}
    for role,summaries in result['summary'].items():
        aggregates[role]={}
        for subset,reported_summary in summaries.items():
            path=directory/f'{role}_{subset}.json'
            saved=json.loads(path.read_text())
            rows=numpy_rows(predictions[role],labels['target'],masks[subset],labels['boundary'],ids)
            summary=numpy_macro(rows)
            check=compare_tree(dict(rows=rows,summary=summary),saved)
            aggregate_check=compare_tree(summary,reported_summary)
            assert check['passed'] and aggregate_check['passed'],(role,subset,check,aggregate_check)
            checks.append(dict(role=role,subset=subset,path=str(path),sha256=sha(path),**check))
            aggregates[role][subset]=summary
    # Load original observation arrays, preserving their native identity order.
    chunks={};oracle={}
    for source in result['source_paths']['val']:
        p=Path(source['path'])
        with np.load(p/'inputs.npz',allow_pickle=False) as data:
            for field in data.files:chunks.setdefault(field,[]).append(data[field].copy())
    arrays={field:np.concatenate(value) for field,value in chunks.items()}
    for source in result['source_paths']['oracle_val']:
        p=Path(source['path']);manifest=json.loads((p/'manifest.json').read_text())
        with np.load(p/'oracle.npz',allow_pickle=False) as data:
            for index,(q,k) in enumerate(manifest['pairs']):
                oracle[(manifest['scene'],q,k)]={field:data[field][index].copy() for field in data.files}
    references={name:[] for name in ('pred_pred','gt_route','gt_value','gt_both')}
    route_mass_error=0.;zero_eligible_pairs=[]
    for index,ident in enumerate(ids):
        o=oracle[identity(ident)]
        q=arrays['q'][index].astype(float);k=arrays['k'][index].astype(float)
        qc=arrays['q_to_kcov'][index].astype(float);kc=arrays['kcov'][index].astype(float)
        assert np.array_equal(kc,np.broadcast_to(kc[:1],kc.shape))
        precision=np.linalg.inv(qc+kc[:1]+.25*np.eye(2))
        delta=arrays['q_to_k'][index].astype(float)[:,None,:]-arrays['kmean'][index].astype(float)[None,:,:]
        energy=-.5*np.einsum('qki,qij,qkj->qk',delta,precision,delta)
        logits=5*(q@k.T)+energy
        weights=np.exp(logits-logits.max(axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
        sensor_weights=np.zeros_like(weights)
        np.add.at(sensor_weights,(np.arange(432)[:,None],o['oracle_indices']),o['oracle_weights'])
        if not o['oracle_valid'].any():zero_eligible_pairs.append(ident)
        route_mass_error=max(route_mass_error,float(np.max(np.abs(sensor_weights.sum(-1)[o['oracle_valid']]-1),initial=0)))
        route=np.where(o['oracle_valid'][:,None],sensor_weights,weights)
        for role,attention,values in [('pred_pred',weights,arrays['k_value_depth'][index]),
                ('gt_route',route,arrays['k_value_depth'][index]),
                ('gt_value',weights,o['k_oracle_value_depth']),('gt_both',route,o['k_oracle_value_depth'])]:
            references[role].append(attention@values.astype(float))
    reference_checks={};reference_predictions={}
    # Predeclared tolerance before this first replay: float64 full energy vs saved
    # float32 quadratic lifting and accumulation, all rows including far projections.
    tolerance_m=.002
    for role,values in references.items():
        actual=np.stack(values);reference_predictions[role]=actual
        error=np.abs(actual-predictions[role])
        maxerror=float(error.max())
        reference_checks[role]=dict(max_absolute_error_m=maxerror,mean_absolute_error_m=float(error.mean()),
            valid_max_absolute_error_m=float(error[labels['valid']].max()),tolerance_m=tolerance_m,passed=maxerror<tolerance_m)
        assert maxerror<tolerance_m,(role,maxerror)
    fallback=~labels['oracle_covisible']
    fallback_same=dict(predicted_values=bool(np.array_equal(predictions['pred_pred'][fallback],predictions['gt_route'][fallback])),
        sensor_values=bool(np.array_equal(predictions['gt_value'][fallback],predictions['gt_both'][fallback])))
    assert all(fallback_same.values())
    output=dict(passed=True,scope='post-run old-validation metric and independent full-kernel reconstruction',
        script_sha256=sha(__file__),transport_summary_sha256=sha(directory/'summary.json'),
        transport_predictions_sha256=sha(directory/'predictions.npz'),lock_sha256=sha(lockpath),
        source_paths=sources,metric_checks=checks,metric_archive_count=len(checks),
        max_metric_absolute_error=max(check['max_absolute_error'] for check in checks),
        independent_kernel_replay=reference_checks,oracle_route_mass_max_error=route_mass_error,
        pairs_with_zero_oracle_eligible_queries=zero_eligible_pairs,
        audit_implementation_note='The first post-run replay stopped on an empty eligible-row max reduction; fixed to an empty-set neutral zero. No experiment, metric population, prediction or frozen source changed.',
        raw_fallback_bitwise_equal=fallback_same,reconstructed_metric_summaries=aggregates,
        model_or_torch_imported=False,any_test_data_read=False,
        caution='Deterministic diagnostic only; native sensor correspondences were selected using depth consistency. No generalization or causal interpretation.')
    dest=Path(__file__).with_name('transport_audit.json')
    assert not dest.exists(),'refuse overwrite'
    dest.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(passed=True,metric_archive_count=len(checks),max_metric_absolute_error=output['max_metric_absolute_error'],
        independent_kernel_replay=reference_checks),indent=2))


if __name__=='__main__':main()
