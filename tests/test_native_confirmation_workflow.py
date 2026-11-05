import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.round65_common import ROOT, VARIANTS, read, sha, write
from experiments.native_confirmation_runtime import summarize_logits
from experiments.native_confirmation_report import (
    collect_exports, aggregate_comparisons, comparison_specs, joint_interval_required,
)
from experiments import native_confirmation_workflow as workflow


class NativeConfirmationWorkflowTests(unittest.TestCase):
    def replication_fixture(self, root, gain=.8):
        write(root / 'queue_status.json', dict(state='complete'))
        write(root / 'lock.json', dict(fixture_only=True))
        selected = dict(all_ten_selected_before_development_evaluation=True,
            records=[dict(role=r, seed=s) for r in ('relative5', 'distance_mlp') for s in (44, 55, 66, 77, 88)])
        write(root / 'selected.json', selected)
        result = dict(lock_sha256=sha(root / 'lock.json'), selection_sha256=sha(root / 'selected.json'),
            new_replication_signal=gain >= .5, cohorts=dict(new5=[dict(variant='base',
                seeds=[44, 55, 66, 77, 88], mean_gain_pp=gain, paired_seed_bootstrap_95_pp=[.1, 1.2])]))
        write(root / 'result.json', result)
        review = dict(scientific_review_complete=True, visual_review_complete=True,
            result_sha256=sha(root / 'result.json'), selection_sha256=sha(root / 'selected.json'),
            lock_sha256=sha(root / 'lock.json'))
        write(root / 'review.json', review)

    def test_incomplete_unreviewed_and_negative_replication_cannot_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(workflow.entry_status(root)['ready'])
            self.replication_fixture(root)
            self.assertTrue(workflow.entry_status(root)['ready'])
            write(root / 'queue_status.json', dict(state='running', stage='train'))
            self.assertFalse(workflow.entry_status(root)['ready'])
            self.replication_fixture(root)
            review = read(root / 'review.json')
            write(root / 'review.json', dict(review, visual_review_complete=False))
            self.assertFalse(workflow.entry_status(root)['ready'])
            write(root / 'review.json', dict(review, result_sha256='wrong'))
            with self.assertRaises(ValueError):
                workflow.entry_status(root)
            self.replication_fixture(root, gain=.4)
            self.assertFalse(workflow.entry_status(root)['ready'])

    def test_unmet_entry_creates_no_seal_or_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            with patch.object(workflow, 'OUT', out), patch.object(workflow, 'checked_protocol',
                    return_value=({}, {'scientific_replication_lock_sha256': 'fixture'})), \
                    patch.object(workflow, 'entry_status', return_value={'ready': False}), \
                    patch.object(workflow, 'population', side_effect=AssertionError('must not prepare test')), \
                    patch.object(workflow, 'MirrorProvider', side_effect=AssertionError('must not fetch')):
                self.assertFalse(workflow.seal()['ready'])
            self.assertFalse(out.exists())

    def test_model_resolution_binds_original_selection_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'best.pt'
            checkpoint.write_bytes(b'fixture checkpoint bytes, not a trained model')
            selection = root / 'selected.json'
            write(selection, dict(records=[dict(role='relative5', seed=11,
                best_epoch=115, checkpoint_sha256=sha(checkpoint))]))
            slot = dict(role='relative5', seed=11, cohort='old3', selection_path=str(selection),
                selection_sha256=sha(selection), checkpoint_path=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), selected_epoch=115)
            self.assertEqual(workflow.resolve_models([slot])[0]['selected_epoch'], 115)
            with self.assertRaises(ValueError):
                workflow.resolve_models([dict(slot, selected_epoch=100)])
            with self.assertRaises(ValueError):
                workflow.resolve_models([slot, slot])
            checkpoint.write_bytes(b'different bytes')
            with self.assertRaises(ValueError):
                workflow.resolve_models([slot])

    def test_export_collection_rejects_wrong_seed_population_or_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, labels = np.array(['a', 'b']), np.array([0, 1])
            groups = {'full_population': np.ones(2, dtype=bool)}
            model = dict(role='relative5', seed=11, path='fixture', sha256='fixture', selected_epoch=5)
            path = root / 'relative5/11/base.npz'
            path.parent.mkdir(parents=True)
            logits = np.zeros((2, 60)); logits[0, 0] = logits[1, 1] = 1
            np.savez(path, ids=ids, labels=labels, logits=logits)
            identity = dict(checkpoint=model, variant='base', cache_identity_sha256='cache',
                cache_manifest_sha256='manifest', runtime_sha256=sha(ROOT / 'experiments/native_confirmation_runtime.py'),
                group_identity={'full_population': hashlib.sha256(groups['full_population'].tobytes()).hexdigest()})
            saved = dict(identity=identity, optimizer_updates=0, prediction_sha256=sha(path),
                         groups=summarize_logits(logits, labels, groups))
            write(path.with_suffix('.json'), saved)
            args = (root, [model], ['base'], ids, labels, groups, 'cache', 'manifest')
            records, predictions = collect_exports(*args)
            self.assertEqual(len(records), 1)
            np.testing.assert_array_equal(predictions['relative5', 11, 'base'], labels)
            with self.assertRaises(ValueError):
                collect_exports(root, [model], ['base'], ids[::-1], labels[::-1], groups, 'cache', 'manifest')
            write(path.with_suffix('.json'), dict(saved, identity=dict(identity, checkpoint=dict(model, seed=22))))
            with self.assertRaises(ValueError):
                collect_exports(*args)
            write(path.with_suffix('.json'), dict(saved, prediction_sha256='wrong'))
            with self.assertRaises(ValueError):
                collect_exports(*args)

    def test_full_comparison_grid_preserves_cohorts_and_joint_scope(self):
        # Small synthetic wiring fixture. It is never a scientific result or a
        # simulated substitute for the 3,845-object confirmation population.
        labels = np.repeat(np.arange(60), 2)
        groups = dict(full_population=np.ones(120, dtype=bool), MACGS=labels < 30,
            ModelNet=labels >= 30, historical_source_reused=(labels >= 30) & (np.arange(120) % 2 == 0),
            no_known_historical_source_overlap=(labels < 30) | (np.arange(120) % 2 == 1))
        specs = comparison_specs()
        keys = {(spec[side], seed, variant) for spec in specs for side in ('left', 'right')
                for seed in spec['seeds'] for variant in VARIANTS}
        self.assertEqual(len(keys), 186)
        records, predictions = [], {}
        for role, seed, variant in sorted(keys):
            prediction = labels.copy()
            if role != 'relative5':
                prediction[(np.arange(120) + seed) % 7 == 0] = (prediction[(np.arange(120) + seed) % 7 == 0] + 1) % 60
            logits = np.zeros((120, 60)); logits[np.arange(120), prediction] = 1
            predictions[role, seed, variant] = prediction
            records.append(dict(role=role, seed=seed, variant=variant, groups=summarize_logits(logits, labels, groups)))
        comparisons = aggregate_comparisons(records, predictions, labels, groups,
            dict(replicates=20, base_seed=76001, chunk_size=8))
        self.assertEqual(len(comparisons), 240)
        self.assertEqual(sum(r['crossed_seed_object_bootstrap_95_pp'] is not None for r in comparisons), 28)
        for row in comparisons:
            self.assertEqual(row['crossed_seed_object_bootstrap_95_pp'] is not None,
                             joint_interval_required(row, row['variant'], row['group']))
            self.assertEqual(len(row['paired_ce_difference']), len(row['seeds']))
            if row['right'] == 'fixed_isotropic5':
                self.assertEqual(row['seeds'], [11, 22, 33])
            if row['cohort'] == 'new5':
                self.assertEqual(row['seeds'], [44, 55, 66, 77, 88])
                self.assertEqual(row['right'], 'distance_mlp')
        damaged = dict(predictions)
        damaged.pop(('distance_mlp', 88, 'base'))
        with self.assertRaises(ValueError):
            aggregate_comparisons(records, damaged, labels, groups, dict(replicates=20, base_seed=76001, chunk_size=8))


if __name__ == '__main__':
    unittest.main()
