"""Acquire a pinned public archive without decoding any point-cloud arrays.

The official test members are retained inside the closed archive. This command
never extracts them. Verified TLS and the published LFS SHA256 are required.
"""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round48_modelnet/acquisition1'
ARCHIVE = ROOT / 'data/round48_modelnet/modelnet40_ply_hdf5_2048.zip'
REVISION = 'd5dc795541800feeb7a4b3bd3142729a0d2adf7a'
EXPECTED_SHA = 'f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7'
EXPECTED_SIZE = 435335111
URL = ('https://huggingface.co/datasets/Msun/modelnet40/resolve/'
       + REVISION + '/modelnet40_ply_hdf5_2048.zip')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists() or ARCHIVE.exists() or ARCHIVE.with_suffix('.part').exists():
        raise ValueError('Preserve existing acquisition; inspect it before retrying.')
    OUT.mkdir(parents=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    plan = dict(
        purpose='Native 3D Transformer PE development, no rendering objective.',
        url=URL, revision=REVISION, expected_sha256=EXPECTED_SHA,
        expected_size=EXPECTED_SIZE, archive=str(ARCHIVE),
        source_sha256=digest(__file__), frozen_unix=time.time(),
        original_source='https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip',
        primary_source='https://github.com/charlesq34/pointnet/blob/master/provider.py',
        mirror_usage='https://github.com/NVlabs/WarpConvNet/blob/main/warpconvnet/dataset/modelnet.py',
        provenance_limit='Published mirror SHA256; no independent original-server byte digest available.',
        decoding_allowed=False, official_test_access=False,
        next_step='Freeze scientific protocol and implementations before decoding official-train HDF5 members.')
    plan_path = OUT / 'plan.json'
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    (OUT/'plan.sha256').write_text(digest(plan_path)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    partial = ARCHIVE.with_suffix('.part')
    count = 0
    h = hashlib.sha256()
    started = time.time()
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('xb') as target:
            for chunk in iter(lambda: response.read(2**20), b''):
                target.write(chunk)
                h.update(chunk)
                count += len(chunk)
                if count % (32*2**20) == 0:
                    print('DOWNLOADED_BYTES', count, flush=True)
        if count != EXPECTED_SIZE or h.hexdigest() != EXPECTED_SHA:
            raise ValueError('Archive size or SHA256 mismatch; partial retained.')
        partial.rename(ARCHIVE)
        with zipfile.ZipFile(ARCHIVE) as archive:
            entries = [dict(name=x.filename, size=x.file_size, crc32=x.CRC)
                       for x in archive.infolist()]
        result = dict(completed=True, archive_sha256=h.hexdigest(), size=count,
                      entries=entries, array_members_opened=[],
                      seconds=time.time()-started, official_test_access=False)
        (OUT/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
        print('ACQUIRED', count, h.hexdigest(), 'members', len(entries), flush=True)
    except BaseException as error:
        # Do not print redirect URLs or authentication details.
        (OUT/'failure.json').write_text(json.dumps(dict(
            error_type=type(error).__name__, downloaded_bytes=count,
            seconds=time.time()-started, partial_preserved=partial.exists()), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
