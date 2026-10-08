from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import residual as core
import model as old

class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path("configs/hai.json").read_text())
        self.rng = np.random.default_rng(11)
        self.x = self.rng.normal(size=(160, 12))

    def test_no_current_target_and_direct_projection(self):
        residual = self.rng.normal(size=(160, 3))
        model = core.fit_correction(self.x, residual, self.cfg)
        prediction = core.predict_correction(self.x, model)
        for target in range(3):
            own, cross = core.feature_indices(target, 3)
            projected = (self.x[:, cross]-model["mean"][cross])-(self.x[:, own]-model["mean"][own]) @ model["projection"][target]
            expected = projected @ model["cross_beta"][target]
            np.testing.assert_allclose(prediction[:, target], expected, atol=1e-10)
            np.testing.assert_allclose(projected.mean(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(projected.T @ (self.x[:, own]-model["mean"][own]), 0, atol=1e-10)
            changed = self.x.copy()
            changed[:, target] += 1000
            np.testing.assert_allclose(core.predict_correction(changed, model)[:, target], prediction[:, target], atol=1e-10)

    def test_pure_self_bias_and_linear_residual_not_misattributed(self):
        residual = np.column_stack([3+self.x[:, i+3]-2*self.x[:, i+6] for i in range(3)])
        model = core.fit_correction(self.x, residual, self.cfg)
        np.testing.assert_allclose(core.predict_correction(self.x, model), 0., atol=1e-11)

    def test_cross_residual_signal_and_nested_zero_option(self):
        residual = np.column_stack([self.x[:, (i+1) % 3]+.05*self.rng.normal(size=160) for i in range(3)])
        model = core.fit_correction(self.x, residual, self.cfg)
        correction = core.predict_correction(self.x, model)
        self.assertTrue(np.all(np.mean((residual-correction)**2, axis=0) < .1*np.mean(residual**2, axis=0)))
        model["weights"][:] = 0
        model["intercept"][:] = 0
        np.testing.assert_array_equal(core.predict_correction(self.x, model), 0.)

    def test_degenerate_own_and_constant_cross_features(self):
        x = np.ones_like(self.x)
        model = core.fit_correction(x, self.rng.normal(size=(160, 3)), self.cfg)
        np.testing.assert_array_equal(core.predict_correction(x, model), 0.)
        np.testing.assert_array_equal(model["own_rank"], 0)

    def test_D_identity_signed_and_zero_correction(self):
        residual = self.rng.normal(size=(100, 3))
        correction = .5*residual
        loss = core.paired_losses(residual, correction, 64)
        np.testing.assert_allclose(loss[:, :, 2], loss[:, :, 1]-loss[:, :, 0], atol=1e-12)
        self.assertTrue(np.all(loss[:, :, 2] < 0))
        np.testing.assert_array_equal(core.paired_losses(residual, 0*residual, 64)[:, :, 2], 0)

    def test_eligibility_requires_both_files_and_block_repeatability(self):
        cfg = dict(self.cfg, channels=3)
        rows = [dict(file=f, stage="cross_file", target=i, es=.5, relative_gain=.06,
                     positive_block_fraction=.7) for f in cfg["roles"]["fit"] for i in range(3)]
        rows[3]["relative_gain"] = -.1
        rows[1]["positive_block_fraction"] = .59
        self.assertEqual([r["eligible"] for r in core.qualify(rows, cfg)], [False, False, True])
        rows[5]["es"] = 1e-9
        self.assertFalse(any(r["eligible"] for r in core.qualify(rows, cfg)))
        rows[0]["file"] = "test4.csv"
        with self.assertRaises(AssertionError):
            core.qualify(rows, cfg)
