"""Describe cached geometric access; this does not measure learned attention."""
import json
from collections import defaultdict

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_bound_data import OUT, CACHE, QUERIES, load_plan, write_json, prior_path
from experiments.gaussian_memory_incremental_analysis import cohorts


def describe(weights, coverage):
    w = weights.double().reshape(-1, weights.shape[-1])
    a = coverage.double().reshape(-1)
    present = a > 0
    count = (w > 0).sum(-1).double()
    p = w[present] / a[present, None]
    effective = (-(p * p.clamp_min(1e-300).log()).sum(-1)).exp()
    return {
        'tokens': len(a), 'coverage_mean': a.mean().item(),
        'zero_coverage_fraction': (~present).double().mean().item(),
        'positive_slots_mean': count.mean().item(),
        'at_most_one_positive_slot_fraction': (count <= 1).double().mean().item(),
        'at_most_four_positive_slots_fraction': (count <= 4).double().mean().item(),
        'effective_slots_entropy_covered_mean': effective.mean().item() if len(p) else None,
        'largest_conditional_prior_covered_mean': p.max(-1).values.mean().item() if len(p) else None,
        'positive_slots_quantiles': torch.quantile(count, torch.tensor([0., .1, .5, .9, 1.], dtype=count.dtype)).tolist(),
    }


def main():
    torch.set_num_threads(4)
    protocol = {
        'diagnostic_only': True, 'all_data_previously_used': True,
        'motivation': 'Interim seed20261112 A/C cost and B/A partial recovery motivate describing the access restriction.',
        'hypothesis': 'The rendering prior can exclude most historical slots even when nonlocal content could be useful.',
        'competing_explanation': 'Restricted access may be appropriate; geometry error, optimization, or state shift can also explain loss.',
        'fixed_comparison': 'All 85 cached initial memories, all four query windows; training12 and primary51 cohorts; no parameter or threshold scan.',
        'measurements': 'Positive-weight slot count, zero coverage, and real-slot conditional prior entropy; query-view and all-window tokens separately.',
        'decision': 'Characterize the hypothesis before choosing a later controlled experiment; cannot change this round, prove nonlocal usefulness, or substitute for matched model outcomes.',
        'interpretation_limit': 'These are geometric prior statistics before content logits, not trained attention weights or information relevance.',
        'query_view_index': -1,
    }
    protocol_path = OUT / 'support_diagnostic_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write_json(protocol_path, protocol)
    plan = load_plan(); rows = []
    for meta in plan['catalog']:
        identity_path = CACHE / meta['id'] / 'identity.json'
        identity = json.loads(identity_path.read_text())
        for query in QUERIES:
            path = prior_path(meta['id'], query)
            assert sha256(path) == identity['sha256'][path.name]
            projected = torch.load(path, map_location='cpu', weights_only=False)
            row = {'id': meta['id'], 'place': meta['place'], 'query': query,
                   'prior_sha256': identity['sha256'][path.name],
                   'prior_identity_sha256': sha256(identity_path)}
            for scope, index in [('query_view', -1), ('all_window_views', slice(None))]:
                row[scope] = describe(projected['weights'][index], projected['coverage'][index])
            rows.append(row)
    result = {'protocol': protocol, 'source_sha256': sha256(__file__), 'rows': rows, 'cohorts': {}}
    for name, subset in cohorts(plan).items():
        ids = {m['id'] for m in subset}; selected = [r for r in rows if r['id'] in ids]
        result['cohorts'][name] = {}
        for scope in ('query_view', 'all_window_views'):
            by_place = defaultdict(list)
            for row in selected:
                by_place[row['place']].append(row[scope])
            keys = [k for k in selected[0][scope] if k not in ('tokens', 'positive_slots_quantiles')]
            summary = {}
            for key in keys:
                values = [[r[key] for r in place_rows if r[key] is not None] for place_rows in by_place.values()]
                nonempty = [float(np.mean(v)) for v in values if v]
                summary[key] = {'place_equal_mean': float(np.mean(nonempty)) if nonempty else None,
                                'places_with_value': len(nonempty)}
            result['cohorts'][name][scope] = summary
    write_json(OUT / 'support_diagnostic.json', result)
    print(json.dumps({name: result['cohorts'][name]['query_view'] for name in ('primary51', 'training12')}, indent=2))


if __name__ == '__main__':
    main()
