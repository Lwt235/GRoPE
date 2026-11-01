import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from experiments import round38_rms_anchor_contract as C


class ContractTests(unittest.TestCase):
    def test_full_population_and_native_order(self):
        entries=[dict(scene=s,pair=[i,i+1],pair_name='p%02d'%i) for s in C.SCENES for i in range(16)]
        specs=C.query_specs(entries)
        self.assertEqual(len(specs),2304);self.assertEqual(len({s['id'] for s in specs}),2304)
        self.assertEqual(specs[0]['query_bounds'],[0,0,24,24]);self.assertEqual(specs[47]['query_bounds'],[168,120,192,144])
        self.assertEqual(specs[48]['query_index'],0)

    def test_duplicate_pair_identity_rejected(self):
        entries=[dict(scene=s,pair=[i,i+1],pair_name='same') for s in C.SCENES for i in range(16)]
        with self.assertRaises(ValueError):C.query_specs(entries)

    def test_no_promotion_scope_and_original_model_budget(self):
        self.assertFalse(C.SCOPE['independent_probability_reference'])
        self.assertFalse(C.SCOPE['scientific_gate_passed']);self.assertFalse(C.SCOPE['goal_complete'])
        self.assertEqual(C.POLICY['max_cell_node_evaluations'],4000000)
        self.assertEqual(C.GATES['production_max_phase_cell_node_evaluations'],9216000000)

    def test_metadata_no_array_or_label_access(self):
        old=C.read(C.ROOT/'results/round20_anchor_fixed/lock.json')
        forbidden={str(Path(e[k]).resolve()) for e in old['entries'] for k in ('label_path','target_path')}
        original=C.sha
        def guarded(path):
            self.assertNotIn(str(Path(path).resolve()),forbidden,'label/target hashed before label stage')
            return original(path)
        with patch.object(C,'sha',side_effect=guarded),patch.object(np,'load',side_effect=AssertionError('metadata decoded NPZ')):
            data=C.metadata()
        self.assertEqual(len(data['entries']),48)
        self.assertTrue(forbidden.isdisjoint(data['bindings']))


if __name__=='__main__':unittest.main()
