import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from model import PairedMLP
from fit_hai import sampled_data
from windows import stratified_indices
from te_replication import new_output
from hai_readouts import gcad_split

class ExecutionSchedule(unittest.TestCase):
    def test_hai_sampling_rng_continuation_and_target_assignment(self):
        cfg=json.loads((ROOT/'configs/hai.json').read_text())
        raw=[np.arange(5000*86,dtype=float).reshape(5000,86)/10000+i for i in range(2)]
        common=dict(mean=np.zeros(86),sd=np.ones(86))
        x,indices,rng=sampled_data(raw,common,cfg)
        old=np.random.default_rng(10)
        expected=[stratified_indices(4984,4096,old) for _ in range(2)]
        np.testing.assert_array_equal(indices,expected)
        order=rng.permutation(len(x)); targets=rng.integers(0,86,len(x))
        np.testing.assert_array_equal(order,old.permutation(len(x)))
        np.testing.assert_array_equal(targets,old.integers(0,86,len(x)))
        net=PairedMLP(86,32,10)
        a=x[order[:256]]; loss,grad=net.loss_grad(a,targets[:256],'self')
        self.assertTrue(np.isfinite(loss)); self.assertTrue(all(np.isfinite(v).all() for v in grad.values()))
    def test_roles_and_whole_family_panel(self):
        c=json.loads((ROOT/'configs/te.json').read_text()); r=c['roles']
        sets=[set(r[n+'_testing_ids']) for n in ['cal_A','cal_B','audit_A','audit_B']]
        self.assertEqual(len(set.union(*sets)),500)
        self.assertEqual(sum(map(len,sets)),500)
        ids=r['fault_panel_A_testing_ids']+r['fault_panel_B_testing_ids']
        self.assertEqual(len(ids),50); self.assertTrue(set(ids)<=sets[2]|sets[3])
    def test_guard_and_nonoverlapping_validation_windows(self):
        with self.assertRaises(ValueError): new_output(ROOT/'results/hai')
        train,val,cut=gcad_split(1000,5)
        self.assertLess(train[-1],val[0]-5); self.assertEqual(cut,800)

if __name__=='__main__': unittest.main()
