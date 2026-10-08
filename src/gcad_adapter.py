"""Official GCAD model with a streaming, equivalent evaluation Jacobian.

The source model is unchanged. BatchNorm is fixed in eval mode and RevIN
statistics are detached exactly as in the author's implementation.
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parents[1]/"vendor/gcad"
sys.path.insert(0, str(VENDOR))
from models.tsmixer import TSMixerRevIN


def build_model(options):
    return TSMixerRevIN((options["history"], 86), 1, options["blocks"],
                       options["dropout"], options["ff_dim"], slice(0, None))


def bn_eval(x, layer):
    scale = layer.weight/torch.sqrt(layer.running_var+layer.eps)
    shift = layer.bias-layer.running_mean*scale
    return x*scale.reshape(x.shape[1:])+shift.reshape(x.shape[1:]), scale.reshape(x.shape[1:])


@torch.no_grad()
def prediction_jacobian(model, x):
    """df_j/dx_(lag,i), [batch, output, lag, input], without 86 backwards."""
    assert not model.training and all(not b.training for b in model.res_blocks)
    rv = model.rev_norm
    mean = x.mean(dim=1, keepdim=True)
    sd = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False)+rv.eps)
    h = (x-mean)/sd*rv.affine_weight+rv.affine_bias
    cache = []
    for block in model.res_blocks:
        n1, s1 = bn_eval(h, block.norm1)
        t1 = F.linear(n1.transpose(1, 2), block.linear1.weight, block.linear1.bias)
        res = h+F.relu(t1).transpose(1, 2)
        n2, s2 = bn_eval(res, block.norm2)
        t2 = F.linear(n2, block.linear2.weight, block.linear2.bias)
        h = res+F.linear(F.relu(t2), block.linear3.weight, block.linear3.bias)
        cache.append((t1 > 0, t2 > 0, s1, s2))
    out = F.linear(h.transpose(1, 2), model.linear.weight, model.linear.bias).transpose(1, 2)
    out = (out-rv.affine_bias)/(rv.affine_weight+rv.eps*rv.eps)*sd+mean
    batch, history, d = x.shape
    g = (torch.eye(d, device=x.device, dtype=x.dtype)[None, :, None, :]
         *model.linear.weight.reshape(1, 1, history, 1)
         *(sd[:, 0]/(rv.affine_weight+rv.eps*rv.eps))[:, :, None, None])
    for block, (mask1, mask2, scale1, scale2) in zip(reversed(model.res_blocks), reversed(cache)):
        # Combine batch/output/lag into a single GEMM row dimension.
        f = (g.reshape(-1, d) @ block.linear3.weight).reshape(batch, d, history, -1)
        f *= mask2[:, None]
        f = (f.reshape(-1, f.shape[-1]) @ block.linear2.weight).reshape(batch, d, history, d)
        gres = g+f*scale2
        t = (gres.transpose(2, 3)*mask1[:, None]) @ block.linear1.weight
        g = gres+t.transpose(2, 3)*scale1
    return out[:, 0], g*rv.affine_weight[None, None, None, :]/sd[:, None]


def sparsify(a, threshold):
    result = torch.relu(a-a.transpose(1, 2))
    result.diagonal(dim1=1, dim2=2).copy_(a.diagonal(dim1=1, dim2=2))
    return torch.where(result < threshold, 0., result)


@torch.no_grad()
def graph_batch(model, x, y, threshold):
    pred, jac = prediction_jacobian(model, x)
    error = pred-y
    a = (jac*(2*error)[:, :, None, None]).abs().mean(dim=2).transpose(1, 2)
    return pred, error.square(), sparsify(a, threshold)


def autograd_graph(model, x, y, threshold):
    """Per-output loss gradients from autograd for independent checks."""
    x = x.detach().clone().requires_grad_(True)
    pred = model(x)[:, 0]
    gradients = []
    for target in range(pred.shape[1]):
        loss = ((pred[:, target]-y[:, target])**2).sum()
        g, = torch.autograd.grad(loss, x, retain_graph=True)
        gradients.append(g.detach().abs().mean(dim=1))
    a = torch.stack(gradients, dim=2)
    return pred.detach(), sparsify(a, threshold)


def dependency_score(a, reference):
    # Public test.py adds epsilon to the reference in BOTH numerator/denominator.
    ref = reference+1e-4
    return ((a-ref).abs()/ref).mean(dim=(1, 2))
