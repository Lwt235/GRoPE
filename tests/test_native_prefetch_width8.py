"""Exercise the permitted eight-slot boundary without any official TEST data."""
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import unittest

from experiments.native_bounded_prefetch import BoundedPrefetch


class WidthEightTests(unittest.TestCase):
    def exercise(self, start):
        payloads = [bytes([i]) * (1024 + i) for i in range(16)]
        rows = [dict(file_path=f'{i}.ply', size=len(data), sha256=hashlib.sha256(data).hexdigest())
                for i, data in enumerate(payloads)]
        barrier = threading.Barrier(8, timeout=10)
        calls, guard = [], threading.Lock()

        def source(row):
            index = int(row['file_path'].split('.')[0])
            with guard:
                calls.append(index)
            if start <= index < start + 8:
                barrier.wait()
            return io.BytesIO(payloads[index])

        with tempfile.TemporaryDirectory() as directory:
            with BoundedPrefetch(rows, source, directory, workers=8) as provider:
                with provider(rows[start]) as stream:
                    self.assertEqual(stream.read(), payloads[start])
                self.assertEqual(set(calls), set(range(start, start + 8)))
                for index in range(start + 1, len(rows)):
                    with provider(rows[index]) as stream:
                        self.assertEqual(stream.read(), payloads[index])
            self.assertEqual(sorted(calls), list(range(start, len(rows))))
            self.assertEqual(provider.snapshot()['peak_active_fetches'], 8)
            self.assertEqual(provider.snapshot()['active_fetches'], 0)
            self.assertTrue(provider.closed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_eight_slots_preserve_order_and_do_not_prefetch_beyond_window(self):
        self.exercise(0)

    def test_resume_at_committed_prefix_avoids_prior_sources(self):
        self.exercise(4)


if __name__ == '__main__':
    unittest.main()
