"""Protect existing GPU jobs while allowing the already authorized sharing."""
import unittest

from experiments.native_confirmation_gpu_resources import eligible_devices, shared_slot_available


class GpuResourceTests(unittest.TestCase):
    def test_prioritize_idle_and_exclude_insufficient_headroom(self):
        def row(index, used, free, util):
            return dict(index=index, uuid=str(index), memory_mib=used,
                        free_mib=free, total_mib=49140, utilization=util)
        rows = [row(0, 10000, 39000, 60), row(1, 100, 48900, 8),
                row(2, 29000, 20000, 0), row(3, 15000, 34000, 30)]
        devices = eligible_devices(rows)
        self.assertEqual([r['index'] for r in devices], [1, 3, 0])
        self.assertEqual([r['shared'] for r in devices], [False, True, True])

    def test_two_shared_jobs_do_not_block_additional_idle_gpus(self):
        active = [(None, {'gpu': {'shared': True}}, None),
                  (None, {'gpu': {'shared': False}}, None)]
        self.assertTrue(shared_slot_available({'shared': True}, active))
        active.append((None, {'gpu': {'shared': True}}, None))
        self.assertFalse(shared_slot_available({'shared': True}, active))
        self.assertTrue(shared_slot_available({'shared': False}, active))


if __name__ == '__main__':
    unittest.main()
