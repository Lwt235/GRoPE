"""24-pixel support extension without editing any sealed Round5/6 source.

Known native footprints and RGB-predicted 3D scatter/continuous-pixel covariance
retain the fixed common reference unit8px. Region labels use the unchanged
Round6 every-source-pixel sensor projection. No future Round7 data are loaded.
"""
from __future__ import annotations
import numpy as np
from experiments import round6_region_data as frozen_data

STRIDES=(8,16,24)
CONDITIONS=((8,24),(24,8),(16,24),(24,16),(24,24))


def pool_features(features,stride):
    if stride in (8,16):
        return frozen_data.pool_features(features,stride)
    if stride!=24:raise ValueError('supported strides are8,16,24')
    features=np.asarray(features)
    if features.ndim!=2 or len(features)!=432 or not np.isfinite(features).all():
        raise ValueError('expected432 finite base8 frozen features')
    blocks=features.astype(np.float64).reshape(18,24,-1).reshape(6,3,8,3,-1)
    pooled=blocks.mean((1,3)).reshape(48,-1)
    return (pooled/np.maximum(np.linalg.norm(pooled,axis=-1,keepdims=True),1e-12)).astype(np.float32)


def prepare_frame(cache,stride):
    if stride in (8,16):return frozen_data.prepare_frame(cache,stride)
    if stride!=24:raise ValueError('unsupported stride')
    depth=np.asarray(cache['predicted_depth'])
    if depth.shape!=(144,192):raise ValueError('full192x144 predicted depth required')
    mean,cov,diagnostics=frozen_data.fit_predicted_support(depth,cache['K'],24)
    return frozen_data.PreparedFrame(pool_features(cache['features'],24),mean,cov,
        frozen_data._K(cache['K']).copy(),frozen_data._pose(cache['c2w']).copy(),24,(144,192),diagnostics)


def make_inputs(query,key,q_stride,k_stride):
    if (q_stride,k_stride) not in CONDITIONS:
        raise ValueError('Round7 extension evaluates only the five NEW24px conditions')
    q=query if isinstance(query,frozen_data.PreparedFrame) else prepare_frame(query,q_stride)
    k=key if isinstance(key,frozen_data.PreparedFrame) else prepare_frame(key,k_stride)
    return frozen_data.make_pair_inputs(q,k,q_stride,k_stride)


def region_labels(query_depth,key_depth,Kq,Kk,c2wq,c2wk,q_stride,k_stride,image_hw=(144,192)):
    if (q_stride,k_stride) not in CONDITIONS:raise ValueError('only new24px conditions')
    return frozen_data.region_supervision(query_depth,key_depth,Kq,Kk,c2wq,c2wk,
        q_stride,k_stride,image_hw=image_hw)


def exact_visible_mass(supervision):
    """576 is not a power of two: do not recover integer mass from rounded f32 ratios."""
    return supervision.target_counts.astype(np.int64).sum(-1)


def oracle_pck(supervision,k_stride,image_hw=(144,192)):
    grid=frozen_data.make_grid(k_stride,image_hw)
    xy=supervision.point_pixcontinuous[supervision.point_gt_valid].astype(np.float64)
    if not len(xy):return {'n_point':0,'pck8_count':0,'pck_stride_count':0}
    distance=np.linalg.norm(xy[:,None]-grid.centers[None],axis=-1).min(-1)
    return {'n_point':len(xy),'pck8_count':int((distance<=8).sum()),
            'pck_stride_count':int((distance<=k_stride).sum())}
