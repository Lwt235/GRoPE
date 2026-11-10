"""Reconcile saved SUN3D candidate names with the second consumed cohort."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the dated inventory.'
    names = {
        'prior': 'results/research_data_usage/sun3d_remaining_metadata_20261107.json',
        'access': 'results/research_data_usage/region_retokenization_confirmation_access_20261109.json',
        'cohort': 'results/region_retokenization_confirmation/preparation/cohort_lock.json',
        'evaluation': 'results/region_retokenization_confirmation/attempt1/evaluation_lock.json',
        'review': 'results/region_retokenization_confirmation/attempt1/review.json',
    }
    records = {k: json.loads((ROOT / v).read_text()) for k, v in names.items()}
    prior = records['prior']
    sources = {str(ROOT / v): digest(ROOT / v) for v in names.values()}
    for group in ('source_sha256', 'checked_historical_source_sha256'):
        for name, expected in prior[group].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            assert digest(path) == expected, str(path)
            sources[str(path)] = expected
    assert records['review']['scientific_review_complete']
    assert records['access']['evaluation_lock_sha256'] == sources[str(ROOT / names['evaluation'])]
    used = {s.split('/')[0] for s in records['access']['scenes']}
    missing = {x['place'] for x in records['access']['metadata_only_missing']}
    selected = set(records['cohort']['places'])
    assert used | missing == selected and not used & missing
    assert set(records['access']['scenes']) == set(records['evaluation']['available_scenes'])
    candidates = set(prior['remaining_directory_candidates'])
    assert selected <= candidates
    excluded = set(prior['excluded_union']) | selected
    remaining = sorted(candidates - selected)
    # Check existing data-disk selection metadata, without reading pixels.
    data_root = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE')
    data_paths = [data_root / n / 'raw/selection.json' for n in (
        'projection_confirmation_20261104', 'region_confirmation_20261105',
        'region_retokenization_confirmation_20261109')]
    for path in data_paths:
        rows = json.loads(path.read_text())['scenes']
        assert {r['scene'].split('/')[0] for r in rows} <= excluded
        sources[str(path)] = digest(path)
    result = dict(checked_unix=time.time(), scope='Saved listings, selection metadata and actual access records only.',
        counts=dict(directory_places=206, prior_excluded=36, additional_used=len(used),
                    additional_metadata_only_missing=len(missing), excluded_union=len(excluded),
                    remaining_directory_candidates=len(remaining)),
        prefixes=prior['prefixes'], excluded_union=sorted(excluded),
        additional_used=sorted(used), additional_metadata_only_missing=sorted(missing),
        remaining_directory_candidates=remaining,
        remaining_by_prefix={p: [x for x in remaining if x.startswith(p+'_')] for p in prior['prefixes']},
        source_sha256=sources, script_sha256=digest(Path(__file__)),
        limitations=['Directory IDs do not prove physical independence or absence of aliases.',
                    'Unselected child scans, calibration and present availability remain unverified.',
                    'The two earlier nine-scene cohorts are consumed; a new metric does not restore freshness.'],
        new_network_requests=0, new_pixel_or_gaussian_content_read=False, new_predictions=0,
        confirmation_cohort_enrolled=False, goal_complete=False)
    assert len(excluded) == 46 and len(remaining) == 160
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT), counts=result['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
