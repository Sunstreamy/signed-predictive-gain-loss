from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
import gcad_adapter as g

class GCADTests(unittest.TestCase):
    def test_official_jacobian_and_asymmetric_graph(self):
        torch.manual_seed(10); torch.set_num_threads(4)
        model = g.build_model(dict(history=5, blocks=6, dropout=0, ff_dim=2048)).double().eval()
        x = torch.randn(2, 5, 86, dtype=torch.float64)
        y = torch.randn(2, 86, dtype=torch.float64)
        pred, raw = g.autograd_graph(model, x, y, .008)
        fast, error, a = g.graph_batch(model, x, y, .008)
        np.testing.assert_allclose(fast.numpy(), pred.numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.numpy(), raw.numpy(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(error.numpy(), (pred.numpy()-y.numpy())**2)
        b = torch.tensor([[[2., 4.], [1., 3.]]])
        np.testing.assert_array_equal(g.sparsify(b, .008), [[[2., 3.], [0., 3.]]])
        reference = torch.tensor([[1., 2.], [0., 4.]])
        expected = ((b.numpy()-(reference.numpy()+1e-4)) / (reference.numpy()+1e-4))
        np.testing.assert_allclose(g.dependency_score(b, reference), np.abs(expected).mean(axis=(1, 2)))
