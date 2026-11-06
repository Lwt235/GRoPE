"""Inventory saved SUN3D directory names against known usage; no data access."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the dated inventory; use a new version for a new observation.'
    paths = {
        'historical': 'results/research_data_usage/confirmation_inventory_20261104.json',
        'region_inventory': 'results/research_data_usage/region_confirmation_inventory_20261105.json',
        'region_access': 'results/research_data_usage/region_confirmation_access_20261105.json',
        'region_selection': 'results/region_confirmation/preparation/scene_selection.json',
        'region_evaluation': 'results/region_confirmation/attempt1/evaluation_lock.json',
        'projection_evaluation': 'results/projection_confirmation/attempt1/evaluation_lock.json',
        'jeffreys_extension': 'results/research_data_usage/region_jeffreys_used_extension_20261105.json',
        'regional_development': 'results/round60_real_scene/attempt1/lock.json',
        'nvs_development': 'results/round70_nvs/attempt1/data/manifest.json',
    }
    records = {key: json.loads((ROOT / name).read_text()) for key, name in paths.items()}
    sources = {name: digest(ROOT / name) for name in paths.values()}
    old, inventory, access = (records[k] for k in ('historical', 'region_inventory', 'region_access'))
    checked_historical = {}
    for name, expected in old['source_sha256'].items():
        actual = digest(ROOT / name)
        assert actual == expected, name
        checked_historical[name] = actual
    assert inventory['historical_inventory_sha256'] == sources[paths['historical']]
    assert inventory['projection_evaluation_lock_sha256'] == sources[paths['projection_evaluation']]
    assert access['evaluation_lock_sha256'] == sources[paths['region_evaluation']]

    root_index = ROOT / 'results/region_confirmation/preparation/sun3d_root_index.html'
    assert digest(root_index) == inventory['root_directory_sha256']
    official = set(re.findall(r'href="((?:brown|harvard|home|hotel|mit)[A-Za-z0-9_-]*)/"', root_index.read_text()))
    assert official == set(inventory['official_places'])
    sources[str(root_index.relative_to(ROOT))] = digest(root_index)
    historical = set(old['excluded_sun3d_places'])
    projection = {s.split('/')[0] for s in records['projection_evaluation']['scenes']}
    regional = set(access['used_places'])
    unavailable = {item['scene'].split('/')[0] for item in access['unavailable_metadata_only']}
    assert historical | projection == set(inventory['excluded_places'])
    assert regional | unavailable == {s.split('/')[0] for s in records['region_selection']['scenes']}
    assert {s.split('/')[0] for s in records['jeffreys_extension']['available_scenes']} <= regional
    regional_old = {item['scene'].split('/')[0] for item in records['regional_development']['entries']}
    nvs_old = {s.split('/')[0] for scenes in records['nvs_development']['scenes'].values() for s in scenes}
    assert regional_old <= historical and nvs_old <= historical
    excluded = historical | projection | regional | unavailable
    assert excluded <= official
    remaining = sorted(official - excluded)
    result = dict(
        checked_unix=time.time(),
        purpose='Establish whether saved metadata retains a pool for a prospective new region cohort.',
        scope='Saved root directory and known usage/selection metadata; not a current server availability probe.',
        counts=dict(directory_places=len(official), historical_exclusions=len(historical),
                    projection_used=len(projection), regional_used=len(regional),
                    unavailable_metadata_only=len(unavailable), excluded_union=len(excluded),
                    remaining_directory_candidates=len(remaining)),
        prefixes=inventory['prefixes'],
        remaining_by_prefix={p: [name for name in remaining if name.startswith(p + '_')]
                             for p in inventory['prefixes']},
        historical_exclusions=sorted(historical), projection_used=sorted(projection),
        regional_used=sorted(regional), unavailable_metadata_only=sorted(unavailable),
        excluded_union=sorted(excluded), remaining_directory_candidates=remaining,
        regional_development_places=sorted(regional_old), nvs_development_places=sorted(nvs_old),
        source_sha256=sources, checked_historical_source_sha256=checked_historical,
        script_sha256=digest(Path(__file__)),
        decision='Retain a metadata candidate pool; choose and freeze any new cohort only after the current development results justify confirmation.',
        limitations=[
            'Historical exclusions include use/preview/reservation records, not 21 newly verified pixel accesses.',
            'Different place names do not establish physical independence or absence of aliases.',
            'Child scans, calibration, frame availability and current network access are not checked for this pool.',
            'A candidate directory is not an enrolled, downloaded or validated confirmation scene.',
            'No new model scores or evidence of task effectiveness are produced.',
        ],
        new_network_requests=0, new_pixel_or_gaussian_content_read=False,
        new_predictions=0, confirmation_cohort_enrolled=False,
        current_training_or_comparisons_changed=False,
    )
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUT), counts=result['counts'],
                         prefix_counts={p: len(names) for p, names in result['remaining_by_prefix'].items()},
                         source_identities_checked=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
