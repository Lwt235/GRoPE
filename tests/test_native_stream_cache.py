"""Check failure and resumption boundaries, not merely helper arithmetic."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.native_stream_cache import build_cache, LocalProvider, stage_object


def fixtures(root):
    fields = ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3',
              'opacity','f_dc_0','f_dc_1','f_dc_2']
    vertices = np.zeros(9, dtype=[(name,'<f4') for name in fields])
    vertices['x'] = np.arange(9)
    vertices['y'] = np.arange(9) % 3
    vertices['z'] = np.arange(9) ** 2
    vertices['rot_0'] = 1
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex 9\n'
              + ''.join('property float ' + name + '\n' for name in fields) + 'end_header\n').encode()
    rows = []
    for index in range(2):
        vertices['opacity'] = index + .5
        payload = header + vertices.tobytes()
        path = root / str(index) / 'sample.ply'
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        rows.append(dict(split='train', taxonomy_id=index, file_path=f'{index}/sample.ply',
                         size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return rows


class StreamCacheTests(unittest.TestCase):
    def test_resume_skips_committed_source_and_cleans_interrupted_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw = root/'original'; rows = fixtures(raw)
            original = {r['file_path']:(raw/r['file_path']).read_bytes() for r in rows}
            provider = LocalProvider(raw); calls=[]
            def open_source(row):
                calls.append(row['file_path'])
                return provider(row)
            def stop(index, progress):
                raise RuntimeError('intentional interruption after first committed row')
            destination = root/'cache'
            with self.assertRaisesRegex(RuntimeError, 'intentional interruption'):
                build_cache(rows, destination, open_source, tokens=8, minimum_free_bytes=0, after_row=stop)
            committed = np.load(destination/'centers.npy')[0].copy()
            # Emulate interruption immediately before staged-copy removal.
            (destination/'raw_stage/raw.verified.ply').write_bytes(original[rows[0]['file_path']])
            build_cache(rows, destination, open_source, tokens=8, minimum_free_bytes=0)
            self.assertEqual(calls, [r['file_path'] for r in rows])
            np.testing.assert_array_equal(np.load(destination/'centers.npy')[0], committed)
            self.assertFalse((destination/'raw_stage/raw.verified.ply').exists())
            self.assertTrue(json.loads((destination/'manifest.json').read_text())['complete'])
            self.assertEqual(original, {r['file_path']:(raw/r['file_path']).read_bytes() for r in rows})
            build_cache(rows, destination, open_source, tokens=8, minimum_free_bytes=0)
            self.assertEqual(len(calls),2)

    def test_wrong_hash_never_commits_and_committed_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); raw=root/'original'; rows=fixtures(raw); destination=root/'cache'
            bad=lambda row:io.BytesIO(b'x'*row['size'])
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                build_cache(rows,destination,bad,tokens=8,minimum_free_bytes=0)
            self.assertEqual(json.loads((destination/'progress.json').read_text())['completed'],[])
            self.assertFalse((destination/'manifest.json').exists())
            self.assertLessEqual((destination/'raw_stage/raw.pending').stat().st_size,rows[0]['size'])
            build_cache(rows,destination,LocalProvider(raw),tokens=8,minimum_free_bytes=0)
            a=np.load(destination/'centers.npy',mmap_mode='r+');a[0,0,0]+=1.;a.flush();del a
            with self.assertRaisesRegex(ValueError,'committed cache row changed'):
                build_cache(rows,destination,LocalProvider(raw),tokens=8,minimum_free_bytes=0)

    def test_truncated_transport_has_finite_retry_and_only_exact_bytes_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder);payload=b'whole old source';row=dict(file_path='0/sample.ply',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            calls=[]
            def source(_):
                calls.append(1)
                return io.BytesIO(payload[:3] if len(calls)<3 else payload)
            with patch('experiments.native_stream_cache.time.sleep'):
                path, errors=stage_object(row,stage,source,0)
            self.assertEqual(len(calls),3);self.assertEqual(len(errors),2)
            self.assertEqual(path.read_bytes(),payload)
            calls.clear()
            with patch('experiments.native_stream_cache.time.sleep'):
                with self.assertRaises(EOFError):
                    stage_object(row,stage,lambda _: (calls.append(1) or io.BytesIO(b'x')),0)
            self.assertEqual(len(calls),3)
            self.assertFalse((stage/'raw.verified.ply').exists())


if __name__=='__main__':
    unittest.main()
