"""Fixed evaluation-only surface support and source construction conditions."""
import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,CACHE,query_path,read
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_memory_incremental_data import write_json
from experiments.gaussian_memory_setup import sha256


def main():
    torch.set_num_threads(4);p=read(OUT/'plan.json');rows=[];source=[]
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];fit=read(CACHE/rid/'source_fit.json');binding=read(OUT/'binding'/f'{rid}.json')
        source.append(dict(id=rid,place=meta['place'],split='development' if meta in p['catalog'] else 'training',
            source_center_rmse_m=fit['registration']['source_center_rmse_m'],
            source_orientation_mean_degrees=fit['registration']['source_rotation_mean_degrees'],
            content_valid=binding['content_valid'],gaussians=binding['gaussian_count'],
            source_state_sha256=sha256(CACHE/rid/'scene_memory.pt')))
        if meta not in p['catalog']:continue
        support_path=CACHE/'evaluation_support'/f'{rid}.pt'
        support=load_tensor(support_path)['labels']
        for q in p['queries']:
            query=load_tensor(query_path(rid,q));valid=query['valid'];roi=valid&query['hidden'];observed=support[str(q)]
            rows.append(dict(id=rid,place=meta['place'],query=q,valid_pixels=int(valid.sum()),roi_valid_pixels=int(roi.sum()),
                supported_pixels=int((valid&observed).sum()),roi_supported_pixels=int((roi&observed).sum()),
                whole_support_fraction=float((valid&observed).sum()/valid.sum()),
                roi_support_fraction=float((roi&observed).sum()/roi.sum()),
                support_file_sha256=sha256(support_path)))
    places=[]
    for meta in p['catalog']:
        group=[row for row in rows if row['id']==meta['id']]
        places.append(dict(id=meta['id'],place=meta['place'],queries=len(group),
            mean_query_roi_support_fraction=float(np.mean([row['roi_support_fraction'] for row in group])),
            mean_query_whole_support_fraction=float(np.mean([row['whole_support_fraction'] for row in group])),
            zero_roi_supported_queries=sum(row['roi_supported_pixels']==0 for row in group)))
    result=dict(complete=True,plan_sha256=sha256(OUT/'plan.json'),queries=rows,places=places,source_conditions=source,
        place_equal_roi_support_fraction=float(np.mean([row['mean_query_roi_support_fraction'] for row in places])),
        place_equal_whole_support_fraction=float(np.mean([row['mean_query_whole_support_fraction'] for row in places])),
        conditions_identical='Sources full in both full/missing queries, fixed ROI identical; label computed only from sensor depths and given cameras',
        evaluation_only=True,no_sample_exclusion=True,no_new_stratification_or_threshold=True,
        scope='Opportunity and source conditions, not evidence of reader/PE utility or a uniquely diagnosed cause')
    write_json(OUT/'source_conditions.json',result)
    print({key:result[key] for key in ('complete','place_equal_roi_support_fraction','place_equal_whole_support_fraction')},flush=True)


if __name__=='__main__':main()
