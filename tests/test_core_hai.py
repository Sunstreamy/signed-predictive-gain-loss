from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import hai_readouts as h

class HAITests(unittest.TestCase):
    def test_split_has_disjoint_raw_support(self):
        train, val, cut = h.gcad_split(101)
        self.assertEqual(train[-1], cut-1)
        self.assertEqual(val[0]-5, cut)
        self.assertEqual(train[0], 5)

    def test_window_alignment(self):
        x = np.arange(100, dtype=float)
        r = h.rolling(x)
        np.testing.assert_array_equal(r, np.array([x[t-63:t+1].mean() for t in range(63, 100)]))
        np.testing.assert_array_equal(h.confirm(r), r[:-2])
        self.assertEqual(len(h.confirm(r)), 100-65)

    def test_tail_ties_and_strict_boundary(self):
        cal = [np.repeat(np.arange(100), 3), np.arange(340)/4]
        values = np.r_[np.arange(400)/4, cal[0], cal[1]]
        expected = np.array([max((1+(c >= v).sum())/(len(c)+1) for c in cal) for v in values])
        np.testing.assert_array_equal(h.tail(values, cal), expected)
        for row in h.cutoffs(cal, [.005, .01, .02]):
            np.testing.assert_array_equal(expected <= row['alpha'], values > row['strict_upper_cutoff'])

    def test_pair_against_direct_linear_solve(self):
        rng = np.random.default_rng(9)
        loss = rng.normal(size=(90, 3, 3))
        cov = np.array([[[2., .5], [.5, 1.]]] * 3)
        fit = dict(mean=np.zeros((3, 2)), sd=np.ones((3, 2)), center=np.ones((3, 2)), precision=np.linalg.inv(cov))
        q, score = h.pair_score(loss, fit)
        direct = np.array([[float((loss[t, j, :2]-1) @ np.linalg.solve(cov[j], loss[t, j, :2]-1))
                            for j in range(3)] for t in range(90)])
        np.testing.assert_allclose(q, direct, rtol=1e-13)
        np.testing.assert_allclose(score, np.array([direct[t-2:t+1].max(axis=1).min() for t in range(2, 90)]))
