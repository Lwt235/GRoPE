"""Synthetic coordinate-frame scope check; no data, weights or tuning.

Fixed diagnostic: compare an analytic Gaussian characteristic function,
explicitly rotated finite frequencies, and frequencies reconstructed from the
rotated mean/covariance using the frozen bank's canonical Cholesky rule.
This is not a training gate or a modification of the frozen R43 experiment.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def run(d):
    rng=np.random.default_rng(20261102)
    eps=rng.normal(size=(6,d));eps-=eps.mean(0)
    eig,vec=np.linalg.eigh(eps.T@eps/len(eps))
    eps=eps@((vec/np.sqrt(eig)[None,:])@vec.T)
    mean=np.arange(1,d+1)*.2
    L=np.diag(np.linspace(.6,1.3,d))
    L[np.tril_indices(d,-1)]=.35
    C=L@L.T
    angle=.73;R=np.eye(d)
    R[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    delta=rng.normal(size=(128,d));rot_delta=delta@R.T
    w=eps@L.T+mean
    rot_mean=R@mean;rot_C=R@C@R.T
    rebuilt=eps@np.linalg.cholesky(rot_C).T+rot_mean
    transformed=w@R.T
    def exact(x,m,c):return np.exp(1j*(x@m)-.5*np.einsum('ni,ij,nj->n',x,c,x))
    def finite(x,omega):return np.exp(1j*(x@omega.T)).mean(-1)
    return dict(dimension=d,queries=len(delta),template_count=len(eps),
        analytic_gaussian_frame_max_error=float(abs(exact(delta,mean,C)-exact(rot_delta,rot_mean,rot_C)).max()),
        explicitly_transformed_frequencies_max_error=float(abs(finite(delta,w)-finite(rot_delta,transformed)).max()),
        cholesky_reconstructed_finite_max_difference=float(abs(finite(delta,w)-finite(rot_delta,rebuilt)).max()),
        reconstructed_mean_max_error=float(abs(rebuilt.mean(0)-rot_mean).max()),
        reconstructed_covariance_max_error=float(abs((rebuilt-rot_mean).T@(rebuilt-rot_mean)/len(eps)-rot_C).max()))


if __name__=='__main__':
    dest=ROOT/'results/round43_frequency/frame_diagnostic.json'
    result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_lock_sha256=(ROOT/'results/round43_frequency/attempt1/lock.sha256').read_text().strip(),
        real_data_read=False,trained_checkpoints_read=False,changes_frozen_training=False,
        results=[run(2),run(3)])
    with dest.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))
