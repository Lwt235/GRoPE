"""Optional symmetric Gaussian query support from observed reconstructed geometry.

This is an explicit input ablation, not a checkpoint or test-derived correction.
The original native rectangle query and the fitted query share RGB and labels.
"""
from pathlib import Path
import numpy as np
import torch


def use_fitted_query(data,metadata,folder,alternate=False):
    suffix='_alt' if alternate else ''
    means,covariances=[],[]
    for pair in metadata['pairs']:
        with np.load(Path(folder)/pair['file'],allow_pickle=False) as f:
            means.append(f['q_projected_mean'+suffix]/24.)
            covariances.append(f['q_projected_cov'+suffix]/24.**2+np.eye(2)*1e-6)
    transformed=dict(data)
    transformed['pixq']=torch.tensor(np.stack(means),dtype=data['pixq'].dtype)
    transformed['projected_qcov']=torch.tensor(np.stack(covariances),dtype=data['projected_qcov'].dtype)
    meta=dict(metadata,query_geometry='projected world Gaussian fitted from independently visible reconstructed GS; no target labels')
    return transformed,meta
