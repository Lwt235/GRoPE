"""Reconcile saved metadata with completed access; no acquisition or scoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'results/research_data_usage/native_candidates_20261106'
INPUTS = dict(
    previous_membership=BASE / 'split_check1/modelnet_membership.json',
    previous_split_summary=BASE / 'split_check1/result.json',
    omni_names=BASE / 'omni_literal_name_mapping.json',
    completed_cross_source=ROOT / 'results/modelnet_cross_source/attempt1/catalog.json',
    completed_cross_source_review=ROOT / 'results/modelnet_cross_source/attempt1/review.json',
    current_development=ROOT / 'results/native_source_diversity/attempt1/catalog.json',
    current_access=ROOT / 'results/research_data_usage/native_source_diversity_access_20261109.json',
    native_release_inventory=ROOT / 'results/research_data_usage/gsmodel60_remaining_metadata_20261107.json',
)


def run():
    assert not (OUT / 'result.json').exists()
    payloads = {name: path.read_bytes() for name, path in INPUTS.items()}
    data = {name: json.loads(content) for name, content in payloads.items()}
    rows = data['previous_membership']['rows']
    assert len(rows) == len({r['model_id'] for r in rows}) == 12309
    known = {r['model_id'] for r in rows if not r['no_known_used_ID']}
    crossed = {r['model_id'] for r in data['completed_cross_source']['rows']}
    development = {r['model_id'] for r in data['current_development']['rows']}
    assert data['completed_cross_source_review']['scientific_review_complete']
    assert len(crossed) == 620 and not (crossed & known)
    assert len(development) == data['current_access']['objects'] == 2657 and development <= known
    assert data['current_access']['prior_historical_use']
    assert not data['current_access']['includes_previous_620_confirmation']
    used = known | crossed | development
    remaining = [r for r in rows if r['model_id'] not in used]
    head_matches = [r for r in remaining if r['exact_head60_taxonomy_id'] is not None]
    primary_classes = {1, 4, 11, 12, 13, 15, 20}
    native = data['native_release_inventory']
    assert native['unassigned_ply_paths'] == 0
    omni = data['omni_names']
    overlap = [r for r in omni['exact_name_matches'] if r['taxonomy_id'] in primary_classes]
    result = dict(observed_unix=time.time(), purpose='Update previously saved object-use metadata, not select a new experiment.',
        sources={name: dict(path=str(INPUTS[name]), sha256=hashlib.sha256(content).hexdigest())
                 for name, content in payloads.items()},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        modelnet=dict(revision=data['completed_cross_source']['revision'], published_objects=len(rows),
            known_used_before_last_confirmation=len(known), previously_new_confirmation_objects_now_used=len(crossed),
            current_native_development_objects_already_used_in_point_form=len(development), known_used_now=len(used),
            remaining_without_known_used_id=len(remaining),
            remaining_by_original_split=dict(Counter(r['upstream_split'] for r in remaining)),
            remaining_by_class=dict(Counter(r['category'] for r in remaining)),
            remaining_exact_head60_matches=head_matches,
            remaining_original_test_with_exact_head60_match=sum(r['upstream_split'] == 'test' for r in head_matches),
            remaining_original_test_in_current_seven=sum(r['upstream_split'] == 'test' and
                r['exact_head60_taxonomy_id'] in primary_classes for r in remaining),
            all_remaining_rows=remaining,
            limitation='Known-ID exclusion within the saved release; not a proof of geometric independence or a statement about future releases.'),
        gsmodel60=dict(revision=native['revision'], published_objects=native['published_ply_paths'],
            unassigned_to_already_used_train_test=0),
        saved_omni_directory=dict(revision=omni['revision'], exact_head60_named_archives=len(omni['exact_name_matches']),
            current_seven_named_archives=overlap, count=len(overlap),
            matched_archive_bytes=sum(r['bytes'] for r in overlap),
            current_class_ids_without_exact_name=sorted(primary_classes - {r['taxonomy_id'] for r in overlap}),
            internal_object_count=None, object_independence_verified=False, semantic_mapping_verified=False,
            interpretation='A different-source candidate directory, not a ready same-source independent replication population.'),
        new_network_requests=0, new_raw_gaussians_read=False, new_model_scores_read=False,
        current_training_or_evaluation_changed=False, new_confirmation_selected=False, goal_complete=False)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('METADATA_USE_RECONCILED', len(used), len(remaining), len(head_matches), len(overlap), flush=True)


if __name__ == '__main__':
    run()
