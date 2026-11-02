"""Prespecified old-validation mechanism diagnostic, no training or selection."""
import hashlib
import numpy as np
import torch
from models.round58_support_pe import scalar_geometry, sine_embedding
from experiments.round58_common import ROOT, OUT, read, write, sha, checked
from experiments.round58_data import load_cache


def main():
    checked()
    destination = OUT/'diagnostics/repeatability'
    if destination.exists():
        raise ValueError('preserve existing diagnostic')
    destination.mkdir(parents=True)
    plan = ROOT/'docs/round58_development_diagnostics.md'
    write(destination/'plan_identity.json', dict(plan_sha256=sha(plan), source_sha256=sha(__file__),
          cache_sha256=read(OUT/'data/val.json')['sha256'], frozen_before_measurement=True,
          changes_training=False, used_confirmation=False))
    cache = load_cache('val')
    names = ('distance', 'gaussian_distance', 'gaussian')
    relative, embedding, per_object = {k: [] for k in names}, {k: [] for k in names}, {k: [] for k in names}
    selected_counts, eligible_counts, matched_counts = [], [], []
    for index, identity in enumerate(cache['ids']):
        centers = torch.from_numpy(cache['centers'][index]).double()
        covariance = torch.from_numpy(cache['covariance'][index]).double()
        euclidean2, metric, shape = scalar_geometry(centers, covariance)
        relations = dict(distance=euclidean2.sqrt().numpy()/.2,
                         gaussian_distance=(.5*metric).sqrt().numpy(),
                         gaussian=(.5*(metric+shape)).sqrt().numpy())
        w = cache['world_centers'][index].astype(np.float64)
        cross_distance = np.linalg.norm(w[0, :, None]-w[1, None], axis=-1)
        overlap = cache['overlap'][index]
        object_errors = {k: [] for k in names}
        for direction in (0, 1):
            d = cross_distance if direction == 0 else cross_distance.T
            o = overlap if direction == 0 else overlap.T
            match = d.argmin(-1)
            rows = np.arange(len(match))
            good = (d[rows, match] <= .1) & (o[rows, match] >= .3)
            i, j = np.triu_indices(len(match), 1)
            allowed = good[i] & good[j] & (match[i] != match[j])
            allowed &= relations['distance'][direction, i, j] >= .5
            allowed &= relations['distance'][1-direction, match[i], match[j]] >= .5
            eligible = np.flatnonzero(allowed)
            seed = int.from_bytes(hashlib.sha256(('R58-repeatability:'+str(identity)+':'+str(direction)).encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            picked = eligible[rng.permutation(len(eligible))[:64]]
            matched_counts.append(int(good.sum()))
            eligible_counts.append(len(eligible))
            selected_counts.append(len(picked))
            if not len(picked):
                continue
            for name in names:
                a = relations[name][direction, i[picked], j[picked]]
                b = relations[name][1-direction, match[i[picked]], match[j[picked]]]
                error = 2*np.abs(a-b)/(a+b)
                pa, pb = sine_embedding(torch.from_numpy(a)).numpy(), sine_embedding(torch.from_numpy(b)).numpy()
                relative[name].append(error)
                embedding[name].append(np.sqrt(np.mean((pa-pb)**2, -1)))
                object_errors[name].append(error)
        for name in names:
            per_object[name].append(float(np.concatenate(object_errors[name]).mean()) if object_errors[name] else np.nan)
    def describe(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                    q10=float(np.quantile(values, .1)), q90=float(np.quantile(values, .9)))
    statistics = {}
    arrays = dict(ids=cache['ids'], selected_counts=np.array(selected_counts), eligible_counts=np.array(eligible_counts),
                  matched_counts=np.array(matched_counts))
    for name in names:
        rel, emb, obj = np.concatenate(relative[name]), np.concatenate(embedding[name]), np.array(per_object[name])
        arrays[name+'_relative'], arrays[name+'_embedding'], arrays[name+'_objects'] = rel, emb, obj
        statistics[name] = dict(relative=describe(rel), embedding_rmse=describe(emb), object_mean_relative=describe(obj))
    paired = {}
    for name in names[1:]:
        diff = np.array(per_object[name])-np.array(per_object['distance'])
        paired[name] = dict(difference=describe(diff), fraction_more_variable=float(np.mean(diff[np.isfinite(diff)] > 0)))
    np.savez_compressed(destination/'values.npz', **arrays)
    result = dict(statistics=statistics, versus_distance=paired, objects=len(cache['ids']),
                  object_directions=len(selected_counts), empty_directions=int((np.array(selected_counts) == 0).sum()),
                  selected_pairs=sum(selected_counts), eligible_pairs=sum(eligible_counts),
                  plan_sha256=sha(plan), source_sha256=sha(__file__), values_sha256=sha(destination/'values.npz'),
                  changes_training=False, task_performance_evidence=False, fresh_confirmation=False)
    write(destination/'summary.json', result)
    print(result)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
