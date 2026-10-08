from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import copy
import qualification as q3
import model as old
import residual

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path("configs/hai.json").read_text())

    def test_qualification_uses_each_fit_file_and_both_checks_only(self):
        cfg = dict(self.cfg, channels=3)
        cross = [dict(target=j, channel=str(j), stage="cross_file", file=f, es=1.,
            relative_gain=.06, positive_block_fraction=.7) for f in cfg["roles"]["fit"] for j in range(3)]
        pooled = [dict(r, stage="pooled_model") for r in cross]
        pooled[0]["relative_gain"] = .049
        pooled[4]["positive_block_fraction"] = .59
        good = q3.qualify_q3(cross, pooled, cfg)
        self.assertEqual([r["q3"] for r in good], [False, False, True])
        extra = [dict(r, file=f, relative_gain=-100) for f in ["train2.csv", "test4.csv"] for r in pooled[:3]]
        self.assertEqual(good, q3.qualify_q3(cross, pooled+extra, cfg))
        broken = copy.deepcopy(cross)
        broken[-1]["es"] = 0
        self.assertFalse(any(r["q3"] for r in q3.qualify_q3(broken, pooled, cfg)))

    def test_signed_loss_aggregation_and_confirmation_order(self):
        rng = np.random.default_rng(21)
        r = rng.normal(size=(90, 3))
        c = .5*r
        loss = residual.paired_losses(r, c)
        mean = np.zeros((3, 3))
        sd = np.arange(1., 10.).reshape(3, 3)
        raw, confirmed, _ = q3.aggregate(loss, mean, sd, 3)
        np.testing.assert_allclose(loss[:, :, 2], loss[:, :, 1]-loss[:, :, 0], atol=1e-12)
        self.assertTrue((loss[:, :, 2] < 0).all())
        for t in range(len(confirmed)):
            np.testing.assert_allclose(confirmed[t], np.min(raw[t:t+3], axis=0))
        extreme = np.zeros((3, 4, 3))
        extreme[:, 3] = 10000
        selected, _, _ = q3.aggregate(extreme[:, :3], mean, sd, 3)
        np.testing.assert_array_equal(selected, 0)

    def test_tail_ties_and_max_over_calibration_files(self):
        values = np.array([[1., 4.], [3., 2.]])
        cal = [np.array([[1., 2.], [1., 4.], [2., 5.]]), np.array([[0., 1.], [3., 4.]])]
        actual = old.branch_pvalues(values, cal)
        expected = [[max((1+sum(c[:, j] >= row[j]))/(len(c)+1) for c in cal) for j in range(2)] for row in values]
        np.testing.assert_array_equal(actual, expected)
