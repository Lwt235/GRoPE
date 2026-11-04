import unittest
import numpy as np
from experiments.round66_classification_diagnostic import compare


class PredictionDecomposition(unittest.TestCase):
    def test_imbalance_can_reverse_oa_and_mca(self):
        labels=np.r_[np.zeros(10,dtype=int),np.arange(1,60)]
        baseline=np.full((len(labels),60),-5.)
        baseline[np.arange(len(labels)),labels]=5.
        baseline[0:2,0]=-5.;baseline[0:2,1]=5.
        candidate=baseline.copy()
        candidate[0:2,0]=5.;candidate[0:2,1]=-5.
        candidate[10,1]=-5.;candidate[10,2]=5.
        r=compare(candidate,baseline,labels)
        self.assertEqual(r['groups']['corrected']['count'],2)
        self.assertEqual(r['groups']['harmed']['count'],1)
        self.assertAlmostEqual(r['differences']['oa'],1/69)
        self.assertAlmostEqual(r['differences']['mca'],(.2-1)/60)

    def test_probability_improvement_without_decision_improvement(self):
        labels=np.arange(60)
        baseline=np.zeros((60,60));candidate=np.zeros((60,60))
        baseline[np.arange(60),(labels+1)%60]=3.
        candidate[np.arange(60),(labels+1)%60]=3.
        candidate[np.arange(60),labels]=2.
        r=compare(candidate,baseline,labels)
        self.assertLess(r['differences']['ce'],0)
        self.assertEqual(r['differences']['oa'],0)
        self.assertEqual(r['groups']['both_wrong']['count'],60)
        back=compare(baseline,candidate,labels)
        self.assertAlmostEqual(r['differences']['ce'],-back['differences']['ce'])


if __name__=='__main__':unittest.main()
