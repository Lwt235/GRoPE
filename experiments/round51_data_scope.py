"""Describe fixed input difficulty; no filtering, selection or model tuning."""
import numpy as np
from experiments.round51_data import OUT, read, write, pixels, sha


def quantiles(values):
    return dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1]))))


def main():
    entries=[]
    for e in read(OUT/'data/manifest.json')['entries']:
        with np.load(e['path'],allow_pickle=False) as z:
            r=z['kgeom'][:,0,:9].reshape(-1,3,3).astype(float)
            angle=np.rad2deg(np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))
            baseline=np.linalg.norm(z['kgeom'][:,0,9:12].astype(float)*3,axis=-1)
            focal_ratio=z['kK'][:,0,0]/z['qK'][:,0,0]
            valid=z['gt_valid'];motion=np.linalg.norm(z['pixcontinuous']-pixels()[None],axis=-1)[valid]
            entries.append(dict(scene=e['scene'],split=e['split'],pairs=len(angle),
                rotation_degrees=quantiles(angle),baseline_metres=quantiles(baseline),
                horizontal_focal_ratio=quantiles(focal_ratio),visible_pixel_displacement=quantiles(motion),
                valid_query_fraction=float(valid.mean()),grid_auc_ceiling=e['argmax_ceiling_auc8']))
    write(OUT/'data_scope.json',dict(entries=entries,source_sha256=sha(__file__),
        scope='Descriptive only: no new strata or claim gates; no fit or evaluation changes.',confirmation_access=False))
    for e in entries:print(e['split'],e['scene'],'rot_med',round(e['rotation_degrees']['median'],2),
        'rot_p90',round(e['rotation_degrees']['p90'],2),'baseline_med',round(e['baseline_metres']['median'],3),
        'focal_ratio',e['horizontal_focal_ratio'],flush=True)


if __name__=='__main__':main()
