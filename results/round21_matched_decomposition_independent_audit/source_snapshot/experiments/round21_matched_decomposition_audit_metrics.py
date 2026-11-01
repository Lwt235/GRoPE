"""Independent R21 reductions: three conditions, ten paired contrasts, no gate.

Only caller-supplied arrays/records are consumed. All metric primitives come
from the frozen independent R20/R15 path, never the R21 producer/model.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numpy as np
from experiments import round20_anchor_audit_metrics as OLD

SCENES = OLD.SCENES
QUERIES = ('g8', 'g16', 'g24')
COUNTS = {q: OLD.COUNTS[q] for q in QUERIES}
VALID_COUNTS = OLD.VALID_COUNTS
CONDITIONS = tuple(c for q in QUERIES for c in OLD.CONDITIONS if c['query'] == q and c['key'] == 'g8')
GEOMETRIES = ('mu_c', 'a_c', 'a_ca', 'full4')
FAMILIES = tuple(g + '_' + s for g in GEOMETRIES for s in ('zero', 'content'))
CONTRASTS = tuple([dict(name='content_' + g, from_role=g + '_zero', to_role=g + '_content') for g in GEOMETRIES]
    + [dict(name=n + '_' + s, from_role=a + '_' + s, to_role=b + '_' + s)
       for n, a, b in (('anchor', 'mu_c', 'a_c'), ('widening', 'a_c', 'a_ca'), ('shape', 'mu_c', 'full4'))
       for s in ('zero', 'content')])
REGION = OLD.REGION
METRICS = OLD.METRICS
fraction_dict, exact_fraction = OLD.fraction_dict, OLD.exact_fraction
integer, number, average = OLD.integer, OLD.number, OLD.average
region_metrics, center_metrics = OLD.region_metrics, OLD.center_metrics


def _cohort(entries):
    return OLD._cohort(entries)


def _identities(rows, entries):
    expected = {(s, pair, q + '_g8') for s, pair in _cohort(entries) for q in QUERIES}
    found = [(r['scene'], tuple(r['pair']), r['condition']) for r in rows]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError('complete 48-pair/three-condition population required')
    return expected


def _means(values, keys):
    values = list(values)
    return {k: average(v[k] for v in values) for k in keys}


def _center_means(values):
    values = list(values)
    return dict(pck=fraction_dict(sum((exact_fraction(v['pck']) for v in values), Fraction()) / len(values)),
                **_means(values, ('mean', 'rmse')))


def summarize_role(regional_rows, center_rows, entries):
    _identities(regional_rows, entries); _identities(center_rows, entries)
    for r in regional_rows:
        q = r['condition'].removesuffix('_g8')
        if (r['query'] != q or r['key'] != 'g8' or r['scenario'] != ('scale' if q == 'g24' else 'partition')
                or integer(r['n_query']) != COUNTS[q] or integer(r['n_key']) != 432):
            raise ValueError('native condition population changed')
        integer(r['mass'])
        for name in METRICS:
            if number(r[name + '_sum']) < -1e-8:
                raise ValueError('negative divergence sum')
    for q in QUERIES:
        subset = [r for r in center_rows if r['condition'] == q + '_g8']
        if sum(integer(r['n_point'], maximum=COUNTS[q]) for r in subset) != VALID_COUNTS[q]:
            raise ValueError('exact original center population changed')
        for r in subset:
            integer(r['pck_count'], maximum=r['n_point'])
            if integer(r['n_query']) != COUNTS[q]:
                raise ValueError('all native query rows required')
    regional, centers = {}, {}
    for scene in SCENES:
        regional[scene], centers[scene] = {}, {}
        for c in CONDITIONS:
            name = c['condition']
            rr = [r for r in regional_rows if r['scene'] == scene and r['condition'] == name]
            mass = sum(r['mass'] for r in rr)
            if mass <= 0:
                raise ValueError('positive original scene/condition mass required')
            regional[scene][name] = dict(mass=mass,
                **{m: math.fsum(number(r[m + '_sum']) for r in rr) / mass for m in METRICS})
            centers[scene][name] = OLD._center_reduction(
                [r for r in center_rows if r['scene'] == scene and r['condition'] == name])
    by_condition = {c['condition']: _means((regional[s][c['condition']] for s in SCENES), METRICS) for c in CONDITIONS}
    by_scene = {s: _means(regional[s].values(), METRICS) for s in SCENES}
    cc = {c['condition']: _center_means(centers[s][c['condition']] for s in SCENES) for c in CONDITIONS}
    cs = {s: _center_means(centers[s].values()) for s in SCENES}
    return dict(by_scene_condition=regional, by_condition=by_condition, by_scene=by_scene,
        macro=_means(by_scene.values(), METRICS),
        center=dict(by_scene_condition=centers, by_condition=cc, by_scene=cs, macro=_center_means(cs.values())))


def paired_center(from_winner, to_winner, bounds, point, valid):
    # The original primitive validates exact winner/mask/target schemas.
    left = center_metrics(from_winner, bounds, point, valid)
    right = center_metrics(to_winner, bounds, point, valid)
    center = (np.asarray(bounds)[:, :2].astype(np.float64) + np.asarray(bounds)[:, 2:] - 1.) / 2
    a, b = center[np.asarray(from_winner)[valid]] - point[valid], center[np.asarray(to_winner)[valid]] - point[valid]
    d2a, d2b = (a * a).sum(-1), (b * b).sum(-1)
    da, db = np.sqrt(d2a), np.sqrt(d2b)
    ca, cb = d2a <= 64., d2b <= 64.
    return dict(n_query=left['n_query'], n_point=left['n_point'], from_correct=left['pck_count'],
        to_correct=right['pck_count'], both_correct=int(np.count_nonzero(ca & cb)),
        both_wrong=int(np.count_nonzero(~ca & ~cb)), benefit=int(np.count_nonzero(~ca & cb)),
        harm=int(np.count_nonzero(ca & ~cb)), flip_valid=int(np.count_nonzero(np.asarray(from_winner)[valid] != np.asarray(to_winner)[valid])),
        flip_all=int(np.count_nonzero(np.asarray(from_winner) != np.asarray(to_winner))),
        distance_delta_sum=math.fsum(map(float, db - da)), squared_distance_delta_sum=math.fsum(map(float, d2b - d2a)),
        from_distance_squared_sum=math.fsum(map(float, d2a)), to_distance_squared_sum=math.fsum(map(float, d2b)),
        rmse_delta=right['rmse'] - left['rmse'] if left['n_point'] else None,
        distance_benefit=int(np.count_nonzero(d2b < d2a)), distance_harm=int(np.count_nonzero(d2b > d2a)),
        distance_equal=int(np.count_nonzero(d2b == d2a)))


def _paired_reduction(rows):
    count_names = ('n_query', 'n_point', 'from_correct', 'to_correct', 'both_correct', 'both_wrong',
                   'benefit', 'harm', 'flip_valid', 'flip_all', 'distance_benefit', 'distance_harm', 'distance_equal')
    result = {n: sum(integer(r[n]) for r in rows) for n in count_names}
    n = result['n_point']
    if (n <= 0 or result['benefit'] - result['harm'] != result['to_correct'] - result['from_correct']
            or sum(result[k] for k in ('both_correct', 'both_wrong', 'benefit', 'harm')) != n
            or sum(result[k] for k in ('distance_benefit', 'distance_harm', 'distance_equal')) != n
            or result['flip_valid'] > n or result['flip_all'] > result['n_query']):
        raise ValueError('inconsistent paired population or transitions')
    result.update({k: math.fsum(number(r[k]) for r in rows) for k in
        ('distance_delta_sum', 'squared_distance_delta_sum', 'from_distance_squared_sum', 'to_distance_squared_sum')})
    result.update(pck_delta=fraction_dict(Fraction(result['to_correct'] - result['from_correct'], n)),
        mean_distance_delta=result['distance_delta_sum'] / n,
        mean_squared_distance_delta=result['squared_distance_delta_sum'] / n,
        rmse_delta=math.sqrt(result['to_distance_squared_sum'] / n) - math.sqrt(result['from_distance_squared_sum'] / n))
    return result


def _paired_means(values):
    values = list(values)
    return dict(pck_delta=fraction_dict(sum((exact_fraction(v['pck_delta']) for v in values), Fraction()) / len(values)),
        **_means(values, ('mean_distance_delta', 'mean_squared_distance_delta', 'rmse_delta')))


def summarize_contrast(rows, entries, declaration):
    _identities(rows, entries)
    for r in rows:
        if (r['contrast'] != declaration['name'] or r['from_role'] != declaration['from_role']
                or r['to_role'] != declaration['to_role']):
            raise ValueError('predeclared contrast identity changed')
    region_keys = tuple(m + '_delta' for m in METRICS)
    regions, centers = {}, {}
    for scene in SCENES:
        regions[scene], centers[scene] = {}, {}
        for c in CONDITIONS:
            name = c['condition']; selected = [r for r in rows if r['scene'] == scene and r['condition'] == name]
            mass = sum(integer(r['mass']) for r in selected)
            if mass <= 0:
                raise ValueError('positive unchanged paired regional mass required')
            regions[scene][name] = dict(mass=mass,
                **{m + '_delta': math.fsum(number(r[m + '_delta_sum']) for r in selected) / mass for m in METRICS})
            centers[scene][name] = _paired_reduction([r['center'] for r in selected])
    bc = {c['condition']: _means((regions[s][c['condition']] for s in SCENES), region_keys) for c in CONDITIONS}
    bs = {s: _means(regions[s].values(), region_keys) for s in SCENES}
    cc = {c['condition']: _paired_means(centers[s][c['condition']] for s in SCENES) for c in CONDITIONS}
    cs = {s: _paired_means(centers[s].values()) for s in SCENES}
    return dict(from_role=declaration['from_role'], to_role=declaration['to_role'], by_scene_condition=regions,
        by_condition=bc, by_scene=bs, macro=_means(bs.values(), region_keys),
        center=dict(by_scene_condition=centers, by_condition=cc, by_scene=cs, macro=_paired_means(cs.values())))


def interactions(contrasts):
    result = {}
    def region_difference(on, off):
        if set(on) != set(off):
            raise ValueError('interaction population differs')
        if 'mass' in on and on['mass'] != off['mass']:
            raise ValueError('interaction regional mass differs')
        return {k: on[k] if k == 'mass' else number(on[k]) - number(off[k])
                for k in on}
    def center_difference(on, off):
        out = dict(pck_delta=fraction_dict(exact_fraction(on['pck_delta']) - exact_fraction(off['pck_delta'])),
            mean_distance_delta=on['mean_distance_delta'] - off['mean_distance_delta'],
            mean_squared_distance_delta=on['mean_squared_distance_delta'] - off['mean_squared_distance_delta'],
            rmse_delta=on['rmse_delta'] - off['rmse_delta'])
        if 'n_point' in on:
            if on['n_point'] != off['n_point']:
                raise ValueError('interaction center population differs')
            out.update(n_point=on['n_point'],
                pck_delta_count=(on['to_correct'] - on['from_correct']) - (off['to_correct'] - off['from_correct']),
                distance_delta_sum=on['distance_delta_sum'] - off['distance_delta_sum'],
                squared_distance_delta_sum=on['squared_distance_delta_sum'] - off['squared_distance_delta_sum'])
        return out
    for name in ('anchor', 'widening', 'shape'):
        on, off = contrasts[name + '_content'], contrasts[name + '_zero']
        out = dict(zero_contrast=name + '_zero', content_contrast=name + '_content', center={})
        for layer in ('by_scene_condition', 'by_condition', 'by_scene', 'macro'):
            if layer == 'by_scene_condition':
                out[layer] = {s: {c: region_difference(on[layer][s][c], off[layer][s][c]) for c in on[layer][s]} for s in SCENES}
                out['center'][layer] = {s: {c: center_difference(on['center'][layer][s][c], off['center'][layer][s][c])
                                              for c in on['center'][layer][s]} for s in SCENES}
            elif layer == 'macro':
                out[layer] = region_difference(on[layer], off[layer])
                out['center'][layer] = center_difference(on['center'][layer], off['center'][layer])
            else:
                out[layer] = {k: region_difference(on[layer][k], off[layer][k]) for k in on[layer]}
                out['center'][layer] = {k: center_difference(on['center'][layer][k], off['center'][layer][k]) for k in on['center'][layer]}
        result[name] = out
    return result
