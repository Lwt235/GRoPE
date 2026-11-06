"""Exercise ordering, interruption recovery, and transport failures end to end."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from experiments.native_bounded_prefetch import BoundedPrefetch
from experiments.native_stream_cache import build_cache, LocalProvider, stage_object
from tests.test_native_stream_cache import fixtures


class PrefetchTests(unittest.TestCase):
    def test_out_of_order_fetch_and_resume_preserve_cache_and_committed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'raw'; rows = fixtures(raw)
            local = LocalProvider(raw)
            original = build_cache(rows, root / 'reference', local, tokens=8, minimum_free_bytes=0)
            second_started = threading.Event()
            def reverse_fetch(row):
                if row == rows[0]:
                    self.assertTrue(second_started.wait(5), 'prefetch did not run concurrently')
                else:
                    second_started.set()
                return local(row)
            def interrupt(index, progress):
                self.assertEqual(index, 0)
                raise RuntimeError('planned interruption after one committed row')
            with BoundedPrefetch(rows, reverse_fetch, root / 'spool', workers=2) as provider:
                with self.assertRaisesRegex(RuntimeError, 'planned interruption'):
                    build_cache(rows, root / 'cache', provider, tokens=8, minimum_free_bytes=0, after_row=interrupt)
                self.assertEqual(provider.snapshot()['peak_active_fetches'], 2)
            self.assertEqual(len(json.loads((root / 'cache/progress.json').read_text())['completed']), 1)
            calls = []
            def resumed_source(row):
                calls.append(row['file_path'])
                return local(row)
            with BoundedPrefetch(rows, resumed_source, root / 'spool', workers=2) as provider:
                result = build_cache(rows, root / 'cache', provider, tokens=8, minimum_free_bytes=0)
            self.assertEqual(calls, [rows[1]['file_path']])
            self.assertEqual(result['array_sha256'], original['array_sha256'])
            for name in ('centers', 'covariance', 'appearance', 'indices', 'labels'):
                np.testing.assert_array_equal(np.load(root / 'reference' / (name + '.npy')),
                                              np.load(root / 'cache' / (name + '.npy')))
            self.assertEqual(list((root / 'spool').iterdir()), [])

    def test_short_response_retries_current_object_without_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixtures(root / 'raw')[:1]
            payload = (root / 'raw' / rows[0]['file_path']).read_bytes(); calls = []
            def source(row):
                calls.append(row['file_path'])
                return io.BytesIO(payload[:3] if len(calls) == 1 else payload)
            stage = root / 'stage'; stage.mkdir()
            with BoundedPrefetch(rows, source, root / 'spool', workers=2) as provider:
                with patch('experiments.native_stream_cache.time.sleep'):
                    path, errors = stage_object(rows[0], stage, provider, 0)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(calls, [rows[0]['file_path']] * 2)
            self.assertEqual([r['error_type'] for r in errors], ['EOFError'])

    def test_wrong_bytes_or_unlisted_identity_cannot_enter_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = fixtures(root / 'raw')
            def corrupt(row):
                return io.BytesIO(b'x' * row['size'])
            with BoundedPrefetch(rows, corrupt, root / 'spool', workers=2) as provider:
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    build_cache(rows, root / 'cache', provider, tokens=8, minimum_free_bytes=0)
                with self.assertRaisesRegex(ValueError, 'frozen catalog'):
                    provider(dict(rows[0], sha256='0' * 64))
            self.assertEqual(json.loads((root / 'cache/progress.json').read_text())['completed'], [])
            self.assertFalse((root / 'cache/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
