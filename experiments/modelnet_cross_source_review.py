"""Independent arithmetic and presentation for the already-sealed transfer test.

No inference, fitting, model selection, new comparisons, or data acquisition.
The original result remains immutable; scientific and visual review is separate.
"""
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from experiments.round65_common import ROOT, read, write, sha

OUT = ROOT/'results/modelnet_cross_source/attempt1'
OLD = ROOT/'results/native_confirmation/attempt1/exports'
CLASSES = np.array([1,4,11,12,13,15,20])


def class_weights(labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    groups = [np.flatnonzero(labels == c) for c in classes]
    matrix = np.zeros((len(labels),len(groups)))
    for column,index in enumerate(groups):
        matrix[index,column] = 1 / len(index)
    return groups,matrix


def independent_pair(left,right,labels,*,seed,replicates=10000):
    """Resample flat object weights and separate accuracy matrices, then subtract."""
    groups,membership = class_weights(labels)
    lc,rc = 100.*(left == labels),100.*(right == labels)
    class_delta = lc @ membership - rc @ membership
    nseeds,nclasses = class_delta.shape
    srng,orng,crng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    draws = {key:[] for key in ('seed','class','object','crossed')}
    for start in range(0,replicates,128):
        n = min(128,replicates-start)
        sw = srng.multinomial(nseeds,np.full(nseeds,1/nseeds),size=n)/nseeds
        cw = crng.multinomial(nclasses,np.full(nclasses,1/nclasses),size=n)/nclasses
        weights = np.zeros((n,len(labels)))
        for index in groups:
            counts = orng.multinomial(len(index),np.full(len(index),1/len(index)),size=n)
            weights[:,index] = counts/(len(index)*nclasses)
        effects = weights @ lc.T - weights @ rc.T
        draws['seed'].append(sw @ class_delta.mean(1))
        draws['class'].append(cw @ class_delta.mean(0))
        draws['object'].append(effects.mean(1))
        draws['crossed'].append(np.einsum('bs,bs->b',sw,effects))
    intervals = {key:np.quantile(np.concatenate(value),[.025,.975]) for key,value in draws.items()}
    return dict(mean=float(class_delta.mean()),seeds=class_delta.mean(1),classes=class_delta.mean(0),
                intervals=intervals)


def independent_source_change(new_left,new_right,new_labels,old_left,old_right,old_labels,
                              *,seed,replicates=10000):
    populations = []
    for left,right,labels in [(new_left,new_right,new_labels),(old_left,old_right,old_labels)]:
        groups,membership = class_weights(labels)
        lc,rc = 100.*(left == labels),100.*(right == labels)
        populations.append((groups,lc,rc,lc @ membership - rc @ membership))
    change = populations[0][3] - populations[1][3]
    nseeds,nclasses = change.shape
    rng = np.random.default_rng(seed)
    draws,class_draws = [],[]
    for start in range(0,replicates,128):
        n=min(128,replicates-start)
        sw=rng.multinomial(nseeds,np.full(nseeds,1/nseeds),size=n)/nseeds
        effects=[]
        for groups,lc,rc,_ in populations:
            weights=np.zeros((n,lc.shape[1]))
            for index in groups:
                counts=rng.multinomial(len(index),np.full(len(index),1/len(index)),size=n)
                weights[:,index]=counts/(len(index)*nclasses)
            effects.append(weights @ lc.T - weights @ rc.T)
        draws.append(np.einsum('bs,bs->b',sw,effects[0]-effects[1]))
        cw=rng.multinomial(nclasses,np.full(nclasses,1/nclasses),size=n)/nclasses
        class_draws.append(cw @ change.mean(0))
    return dict(mean=float(change.mean()),seeds=change.mean(1),classes=change.mean(0),
                interval=np.quantile(np.concatenate(draws),[.025,.975]),
                class_interval=np.quantile(np.concatenate(class_draws),[.025,.975]))


def load_population(lock,*,old):
    catalog = (ROOT/'results/native_confirmation_metadata/attempt1/test_catalog.json' if old else OUT/'catalog.json')
    rows=read(catalog)['rows']
    labels=np.array([r['taxonomy_id'] for r in rows])
    ids=np.array([r['file_path'] for r in rows])
    mask=np.isin(labels,CLASSES) if old else np.array([not r['known_byte_reuse'] for r in rows])
    assert np.array_equal(np.unique(labels[mask]),CLASSES)
    expected=read(OUT/('old_source_reference.json' if old else 'result.json'))
    saved={(r['role'],r['seed']):r for r in expected['records']}
    predictions,errors,identities={},[],set()
    for model in lock['models']:
        key=model['role'],model['seed']
        path=(OLD if old else OUT/'evaluation')/key[0]/str(key[1])/'base.npz'
        sidecar=read(path.with_suffix('.json'))
        assert sha(path)==sidecar['prediction_sha256']==saved[key]['prediction_sha256']
        assert sidecar['identity']['checkpoint']==model and sidecar['identity']['variant']=='base'
        assert sidecar['optimizer_updates']==0
        identities.add((sidecar['identity']['cache_identity_sha256'],sidecar['identity']['cache_manifest_sha256']))
        with np.load(path,allow_pickle=False) as archive:
            assert np.array_equal(archive['ids'],ids) and np.array_equal(archive['labels'],labels)
            logits=np.asarray(archive['logits'],dtype=np.float64)
        assert logits.shape==(len(labels),60) and np.isfinite(logits).all()
        x,y=logits[mask],labels[mask]
        pred=x.argmax(1);correct=pred==y
        counts=np.bincount(y,minlength=60);wins=np.bincount(y,weights=correct,minlength=60)
        class_accuracy=wins[counts>0]/counts[counts>0]
        metrics=dict(mca=float(class_accuracy.mean()),oa=float(correct.mean()),
                     ce=float((logsumexp(x,axis=1)-x[np.arange(len(y)),y]).mean()))
        for metric,value in metrics.items():
            errors.append(abs(value-saved[key]['metrics'][metric]))
        errors.extend(np.abs(class_accuracy-np.array(saved[key]['metrics']['class_accuracy'])[CLASSES]).tolist())
        errors.append(abs(float((~np.isin(pred,CLASSES)).mean())-saved[key]['outside_seven_fraction']))
        predictions[key]=pred
    assert len(identities)==1 and len(predictions)==19
    return predictions,labels[mask],max(errors),identities


def review():
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():
        assert sha(path)==digest,path
    for model in lock['models']:
        assert sha(model['path'])==model['sha256']
    result=read(OUT/'result.json');reference=read(OUT/'old_source_reference.json')
    assert result['lock_sha256']==reference['lock_sha256']==sha(OUT/'lock.json')
    populations={};errors=[];comparison_checks=[];exchanges=[]
    for old,key,summary in [(True,'old',reference),(False,'new',result)]:
        predictions,labels,error,_=load_population(lock,old=old)
        populations[key]=(predictions,labels);errors.append(error)
        for comparison in summary['comparisons']:
            left=np.stack([predictions['relative5',s] for s in comparison['seeds']])
            right=np.stack([predictions[comparison['right'],s] for s in comparison['seeds']])
            checked=independent_pair(left,right,labels,seed=comparison['bootstrap_seed'])
            errors += [abs(checked['mean']-comparison['mean_gain_pp']),
                       float(np.abs(checked['seeds']-comparison['paired_seed_gains_pp']).max()),
                       float(np.abs(checked['classes']-comparison['class_mean_gain_pp']).max())]
            for short,long in [('seed','paired_seed_bootstrap_95_pp'),('class','paired_class_bootstrap_95_pp'),
                               ('object','within_class_object_bootstrap_95_pp'),('crossed','crossed_seed_object_bootstrap_95_pp')]:
                errors.append(float(np.abs(checked['intervals'][short]-comparison[long]).max()))
            comparison_checks.append(dict(population=key,right=comparison['right'],cohort=comparison['cohort']))
            _,membership=class_weights(labels)
            corrected=(left==labels)&(right!=labels);regressed=(left!=labels)&(right==labels)
            correct_rate=100.*(corrected @ membership).mean(1)
            regress_rate=100.*(regressed @ membership).mean(1)
            assert np.allclose(correct_rate-regress_rate,checked['seeds'],atol=1e-10)
            exchanges.append(dict(population=key,right=comparison['right'],cohort=comparison['cohort'],
                seeds=comparison['seeds'],corrected_rate_pct=correct_rate.tolist(),regressed_rate_pct=regress_rate.tolist(),
                corrected_counts=corrected.sum(1).tolist(),regressed_counts=regressed.sum(1).tolist()))
    new,new_labels=populations['new'];old,old_labels=populations['old']
    for row in result['source_gain_changes']:
        arrays=[np.stack([pop[role,s] for s in row['seeds']])
                for pop in (new,old) for role in ('relative5',row['right'])]
        checked=independent_source_change(arrays[0],arrays[1],new_labels,arrays[2],arrays[3],old_labels,
                                          seed=row['bootstrap_seed'])
        errors += [abs(checked['mean']-row['mean_gain_change_pp']),
                   float(np.abs(checked['seeds']-row['paired_seed_gain_changes_pp']).max()),
                   float(np.abs(checked['classes']-row['class_mean_gain_changes_pp']).max()),
                   float(np.abs(checked['interval']-row['crossed_seed_object_bootstrap_95_pp']).max()),
                   float(np.abs(checked['class_interval']-row['class_sensitivity_bootstrap_95_pp']).max())]
    primary=result['primary']
    expected=primary['mean_gain_pp']>=.5 and primary['crossed_seed_object_bootstrap_95_pp'][0]>0
    assert expected==result['primary_limited_signal']
    assert primary['right']=='distance_mlp' and primary['seeds']==[11,22,33,44,55,66,77,88]
    assert result['training_updates']==0 and result['goal_complete'] is False
    maximum=max(errors)
    if maximum>1e-8:
        raise ValueError('material independent arithmetic disagreement: '+str(maximum))
    write(OUT/'independent_checks.json',dict(passed=True,completed_unix=time.time(),maximum_error=maximum,
        comparisons=comparison_checks,source_change_comparisons=2,models=19,old_objects=len(old_labels),
        new_primary_objects=len(new_labels),result_sha256=sha(OUT/'result.json'),
        implementation_sha256=sha(__file__),lock_sha256=sha(OUT/'lock.json'),error_exchanges=exchanges,
        scientific_review_pending=True,visual_review_pending=True,goal_complete=False))
    return result,reference


def plot(result,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.4),layout='constrained')
    titles=['Old source: 422 MACGS objects','New source: ModelNet objects','Change in PE gain: new - old']
    limits=[]
    for axis,title,data,change in zip(axes,titles,[reference['comparisons'][:2],result['comparisons'][:2],result['source_gain_changes']],[False,False,True]):
        for position,row in enumerate(data):
            value=row['mean_gain_change_pp' if change else 'mean_gain_pp']
            lo,hi=row['crossed_seed_object_bootstrap_95_pp'];limits.extend([lo,hi])
            color=['#2463a8','#b76c28'][position]
            axis.plot([lo,hi],[position,position],color=color,lw=2)
            axis.plot(value,position,'o',color=color,ms=7)
            axis.annotate(f'{value:+.2f}',(value,position),xytext=(0,10),textcoords='offset points',ha='center')
        axis.axvline(0,color='.5',lw=1,ls='--');axis.set_title(title,fontsize=10)
        axis.set_yticks([0,1],['F - Distance (8 seeds)','F - fixed sphere (3 seeds)'])
        axis.set_ylim(1.7,-.7);axis.set_xlabel('Class-balanced accuracy difference (pp)')
        axis.grid(axis='x',alpha=.15)
    span=max(limits)-min(limits);padding=max(1.,span*.08)
    for axis in axes:axis.set_xlim(min(limits)-padding,max(limits)+padding)
    fig.savefig(OUT/'source_effects.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    result,reference=review()
    plot(result,reference)
    print('INDEPENDENT_REVIEW_READY',read(OUT/'independent_checks.json')['maximum_error'],flush=True)
