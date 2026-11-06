"""Bounded raw-byte prefetch for the unchanged sequential native cache builder."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import time


class BoundedPrefetch:
    """Fetch up to `workers` catalog objects; one caller consumes in catalog order.

    Returned files are owned by the caller's normal `with provider(row)` block.
    TemporaryFile removes only files created here, including on process exit.
    The original stager still independently checks full size and SHA256.
    """

    def __init__(self, rows, provider, spool, *, workers=4, minimum_free_bytes=0):
        self.rows = list(rows)
        self.lookup = {row['file_path']: i for i, row in enumerate(self.rows)}
        if not self.rows or len(self.lookup) != len(self.rows) or not 1 <= workers <= 8:
            raise ValueError('invalid population or prefetch width')
        if any(row['size'] <= 0 or len(row['sha256']) != 64 for row in self.rows):
            raise ValueError('invalid object identity')
        self.provider, self.workers = provider, workers
        self.spool = Path(spool)
        if self.spool.is_symlink():
            raise ValueError('prefetch spool must not be a symlink')
        self.spool.mkdir(parents=True, exist_ok=True)
        self.reserve = minimum_free_bytes
        self.maximum_object_bytes = max(row['size'] for row in self.rows)
        self.maximum_auxiliary_raw_bytes = workers * self.maximum_object_bytes
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native-prefetch')
        self.futures = {}
        self.last_index = None
        self.closed = False
        self.stats = dict(started_unix=time.time(), workers=workers, fetch_attempts=0,
                          completed_fetches=0, fetched_bytes=0, delivered_objects=0,
                          active_fetches=0, peak_active_fetches=0)

    def _fetch(self, index):
        row = self.rows[index]
        if self.stop.is_set():
            raise InterruptedError('prefetch stopped')
        # Conservative reserve covers this whole window plus the original stager.
        required = self.reserve + (self.workers + 1) * self.maximum_object_bytes
        if shutil.disk_usage(self.spool).free < required:
            raise OSError('insufficient disk reserve for bounded prefetch')
        with self.guard:
            self.stats['fetch_attempts'] += 1
            self.stats['active_fetches'] += 1
            self.stats['peak_active_fetches'] = max(self.stats['peak_active_fetches'], self.stats['active_fetches'])
        target = None
        try:
            target = tempfile.TemporaryFile(mode='w+b', dir=self.spool)
            digest, size = hashlib.sha256(), 0
            with self.provider(row) as source:
                while True:
                    if self.stop.is_set():
                        raise InterruptedError('prefetch stopped')
                    chunk = source.read(2**20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > row['size']:
                        raise ValueError('prefetched source exceeds declared size')
                    target.write(chunk)
                    digest.update(chunk)
                    with self.guard:
                        self.stats['fetched_bytes'] += len(chunk)
            if size != row['size']:
                raise EOFError('prefetched source shorter than declared size')
            if digest.hexdigest() != row['sha256']:
                raise ValueError('prefetched source SHA256 mismatch')
            target.flush()
            target.seek(0)
            with self.guard:
                self.stats['completed_fetches'] += 1
            return target
        except BaseException:
            if target is not None:
                target.close()
            raise
        finally:
            with self.guard:
                self.stats['active_fetches'] -= 1

    def __call__(self, row):
        if self.closed:
            raise RuntimeError('prefetch provider is closed')
        index = self.lookup.get(row['file_path'])
        if index is None or row != self.rows[index]:
            raise ValueError('request differs from frozen catalog')
        if self.last_index is not None and index not in (self.last_index, self.last_index + 1):
            raise ValueError('consumer must proceed in catalog order, allowing current-row retry')
        self.last_index = index
        # A first request can start at any committed-prefix boundary on resumption.
        for candidate in range(index, min(len(self.rows), index + self.workers)):
            if candidate not in self.futures:
                self.futures[candidate] = self.pool.submit(self._fetch, candidate)
        future = self.futures[index]
        # If fetching fails, the unchanged stager can request this same row again.
        try:
            result = future.result()
        except Exception:
            self.futures.pop(index)
            raise
        # Keep a pending future tracked if the main thread receives SystemExit or
        # KeyboardInterrupt, so close() can reclaim its eventual temporary file.
        self.futures.pop(index)
        with self.guard:
            self.stats['delivered_objects'] += 1
        return result

    def snapshot(self):
        with self.guard:
            return dict(self.stats, maximum_auxiliary_raw_bytes=self.maximum_auxiliary_raw_bytes,
                        original_stager_additional_maximum_bytes=self.maximum_object_bytes,
                        last_requested_index=self.last_index, closed=self.closed)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for future in self.futures.values():
            if not future.cancelled():
                try:
                    future.result().close()
                except BaseException:
                    pass
        self.futures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
