"""Account for new-write RGB information already present in the native window."""
import json

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_soft_data import OUT, BASE_CACHE, QUERIES, load_plan, write_json


def main():
    plan = load_plan()
    protocol = {
        'hypothesis': 'Most newly written RGB views are also visible to the native four-view query encoder, limiting the new information supplied by incremental memory.',
        'competing_explanations': ['Memory organization can help even with overlapping RGB.',
            'Geometry noise, frozen-backbone adaptation and static-versus-dynamic training mismatch remain unresolved.'],
        'fixed_comparison': 'For every existing catalog clip and every query condition, compare prior writable-frame indices with the query encoder input-frame indices; record set differences.',
        'scope': 'Input identity accounting only; frame overlap is not a measure of useful pixels, geometry quality, attention or downstream benefit.',
        'decision': 'If overlap dominates, a subsequent experiment should explicitly expose history outside the native input window or extend the history before drawing general incremental-memory conclusions. Do not modify the current study.',
        'created_before_accounting_results': True, 'new_training_or_tuning': False,
        'plan_sha256': sha256(OUT / 'plan.json'), 'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_soft_information.py')}
    protocol_path = OUT / 'information_accounting_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text()) == protocol
    else: write_json(protocol_path, protocol)
    rows = []; identities = {}; by_query = {str(q): [] for q in QUERIES}
    for meta in plan['catalog']:
        rid = meta['id']; path = BASE_CACHE / rid / 'identity.json'
        identity = json.loads(path.read_text()); identities[rid] = sha256(path)
        queries = [d for d in identity['diagnostics'] if d['kind'] == 'query']
        assert len(queries) == 12
        for d in queries:
            q = d['window'][-1]; writable = {f for f in plan['write_frames'] if f < q}
            context = set(d['window'][:-1]); assert q in QUERIES and all(f < q for f in context)
            assert not context.intersection(QUERIES)
            record = {'id': rid, 'query': q, 'condition': d['condition'], 'native_input_frames': d['window'],
                      'new_write_frames_available': sorted(writable), 'new_writes_in_native_context': sorted(writable & context),
                      'new_writes_outside_native_context': sorted(writable - context),
                      'initial_reference_frames_outside_native_context': sorted(set(plan['reference']) - context)}
            rows.append(record)
            if d['condition'] == 0: by_query[str(q)].append(record)
    summary = {}
    for q, values in by_query.items():
        keys = ('native_input_frames', 'new_write_frames_available', 'new_writes_in_native_context',
                'new_writes_outside_native_context', 'initial_reference_frames_outside_native_context')
        variants = {key: sorted({tuple(r[key]) for r in values}) for key in keys}
        assert len(values) == len(plan['catalog'])
        summary[q] = {'clips': len(values), 'distinct_frame_sets': variants,
                      'mean_new_writes_outside_native_context': sum(len(r['new_writes_outside_native_context']) for r in values) / len(values)}
    result = {'complete': True, 'clips': len(plan['catalog']), 'query_condition_records': len(rows),
              'all_input_identities': identities, 'by_query': summary, 'rows': rows,
              'scope': protocol['scope'], 'all_data_previously_used': True, 'scientific_settings_changed': False}
    write_json(OUT / 'information_accounting.json', result)
    print(json.dumps({'clips': result['clips'], 'records': len(rows), 'by_query': summary}), flush=True)


if __name__ == '__main__': main()
