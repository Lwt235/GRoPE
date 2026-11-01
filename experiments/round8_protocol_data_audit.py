"""Independent NumPy contract audit of old-development oracle archives."""
from pathlib import Path
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'harvard_c3/hv_c3_1': ('train', 192),
    'harvard_c5/hv_c5_1': ('train', 192),
    'harvard_c6/hv_c6_1': ('train', 192),
    'harvard_c8/hv_c8_3': ('val', 64),
    'brown_bm_4/brown_bm_4': ('val', 64),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT/'results/round8_oracle_data'
    destination = ROOT/'results/round8_protocol_audit/verification.json'
    if destination.exists():
        raise ValueError('refuse overwrite '+str(destination))
    protocol = json.loads((source/'frozen_export_protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items():
        assert sha(ROOT/name) == digest == sha(source/'source'/name), name
    reports = []
    for scene, (split, count) in ALLOWED.items():
        folder = source/split/scene.replace('/', '__')
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['scene'] == scene and manifest['split'] == split
        assert len(manifest['pairs']) == count
        assert manifest['oracle_sha256'] == sha(folder/'oracle.npz')
        base = Path(manifest['base_path'])
        assert base.parent.name == split and 'test' not in base.parts
        assert sha(base/'manifest.json') == manifest['base_manifest_sha256']
        assert sha(base/'inputs.npz') == manifest['base_input_sha256']
        assert sha(base/'labels.npz') == manifest['base_label_sha256']
        with np.load(folder/'oracle.npz', allow_pickle=False) as raw:
            data = {key: raw[key] for key in raw.files}
        with np.load(base/'labels.npz', allow_pickle=False) as raw:
            valid = raw['valid']
        with np.load(base/'inputs.npz', allow_pickle=False) as raw:
            baseline = raw['q_depth']
        eligible = data['oracle_valid']
        assert eligible.shape == valid.shape == (count, 432)
        assert not (eligible & ~valid).any()
        assert np.array_equal(eligible, data['oracle_status'] == 0)
        weights, indices = data['oracle_weights'], data['oracle_indices']
        assert weights.shape == indices.shape == (count, 432, 4)
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert ((indices >= 0) & (indices < 432)).all()
        assert not weights[~eligible].any()
        sums = weights.astype(np.float64).sum(-1)
        norm_error = float(np.max(np.abs(sums[eligible]-1)))
        assert norm_error < 1e-7
        # Independently reconstruct the spatial barycenter from sparse indices.
        x = (indices % 24)*8+3.5
        y = (indices // 24)*8+3.5
        center = np.stack(((x*weights).sum(-1), (y*weights).sum(-1)), -1)
        pixels = data['oracle_pixel']
        expected = np.clip(pixels, (3.5, 3.5), (187.5, 139.5))
        center_error = float(np.max(np.abs(center[eligible]-expected[eligible])))
        assert center_error < 2e-5
        clipped = eligible & ((pixels[..., 0]<3.5) | (pixels[..., 0]>187.5)
                              | (pixels[..., 1]<3.5) | (pixels[..., 1]>139.5))
        assert np.array_equal(clipped, data['route_clipped'])
        fraction = data['k_sensor_fraction']
        assert fraction.shape == eligible.shape and np.isfinite(fraction).all()
        assert ((fraction >= 0) & (fraction <= 1)).all()
        assert np.array_equal(fraction*64, np.rint(fraction*64))
        value = data['k_oracle_value_depth']
        assert np.isfinite(value).all() and ((value >= .1) & (value <= 20)).all()
        for field in ('direct_corresponding_pred_depth', 'direct_corresponding_sensor_depth'):
            assert np.isfinite(data[field]).all() and (data[field] > 0).all()
            assert np.array_equal(data[field][~eligible], baseline[~eligible])
        reports.append(dict(scene=scene, split=split, pairs=count,
            query_valid_count=int(valid.sum()), eligible_count=int(eligible.sum()),
            fallback_query_valid_count=int((valid & ~eligible).sum()),
            clipped_eligible_count=int(clipped.sum()),
            eligible_fraction_of_query_valid=float(eligible.sum()/valid.sum()),
            route_normalization_max_error=norm_error,
            route_barycenter_max_pixel_error=center_error,
            oracle_sha256=sha(folder/'oracle.npz')))
    result = dict(passed=True, scope='all 704 old-development pairs; no fresh test data',
        protocol_sha256=sha(source/'frozen_export_protocol.json'),
        audit_source_sha256=sha(__file__), reports=reports,
        checks=['unchanged base array hashes', 'exact fixed development population',
            'eligibility subset of query labels and status consistency',
            'sparse route normalization and independent pixel barycenter',
            'exact edge-clamp flags', 'sensor fractions in 64-pixel increments',
            'finite bounded oracle values', 'exact query fallback for both direct transports'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
