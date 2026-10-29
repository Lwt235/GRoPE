"""Open-set matching metrics; visibility never enters model features."""
import numpy as np
import torch
import torch.nn.functional as F


def average_precision(labels,scores):
    labels=np.asarray(labels,dtype=bool)
    if not labels.any(): return float('nan')
    order=np.argsort(-scores,kind='stable')
    y=labels[order]
    sorted_scores=np.asarray(scores)[order]
    # Group equal thresholds so constant predictions have prevalence AP.
    ends=np.r_[np.flatnonzero(np.diff(sorted_scores)!=0),len(y)-1]
    tp=np.cumsum(y)[ends]
    precision=tp/(ends+1)
    recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def metrics(probability,labels,pixel_error):
    p=np.asarray(probability)
    labels=np.asarray(labels)
    n=p.shape[-1]-1
    pred=p.argmax(-1)
    visible=labels<n
    correct=pred==labels
    accepted=pred<n
    matched_correct=correct&visible
    per_scene=(matched_correct.sum(-1)/np.maximum(visible.sum(-1),1))
    confidence=p.max(-1)
    ece=0.
    for low in np.linspace(0,1,11)[:-1]:
        mask=(confidence>=low)&(confidence<low+.1 if low<.9 else confidence<=1)
        if mask.any(): ece+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    picked=np.take_along_axis(pixel_error,np.minimum(pred,n-1)[...,None],axis=-1)[...,0]
    summary={'visible_r1':float(matched_correct.sum()/max(visible.sum(),1)),
             'visible_r1_scene_mean':float(per_scene.mean()),
             'overall_accuracy':float(correct.mean()),
             'match_precision':float(matched_correct.sum()/max(accepted.sum(),1)),
             'match_recall':float(matched_correct.sum()/max(visible.sum(),1)),
             'unmatched_auprc':average_precision((~visible).flatten(),p[...,-1].flatten()),
             'unmatched_prevalence':float((~visible).mean()),
             'false_match_rate':float((accepted&~visible).sum()/max((~visible).sum(),1)),
             'ece':float(ece),
             'brier_unmatched':float(((p[...,-1]-(~visible))**2).mean()),
             'accepted_visible_pixel_error':float(picked[visible&accepted].mean()) if (visible&accepted).any() else None}
    return summary,per_scene


@torch.no_grad()
def evaluate(model,data,batch=32):
    probability=[]
    losses=[]
    for start in range(0,len(data['q']),batch):
        part={k:v[start:start+batch] for k,v in data.items()}
        logits=model(part)
        assert torch.isfinite(logits).all()
        probability.append(logits.softmax(-1).cpu().numpy())
        losses.append(F.cross_entropy(logits.flatten(0,1),part['labels'].flatten(),reduction='none').cpu())
    p=np.concatenate(probability)
    summary,scenes=metrics(p,data['labels'].cpu().numpy(),data['pixel_error'].cpu().numpy())
    summary['loss']=float(torch.cat(losses).mean())
    return summary,scenes,p
