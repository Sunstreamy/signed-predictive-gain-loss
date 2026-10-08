import numpy as np
from windows import confirmed_score, rolling_mean


def lag_major(raw, mean, sd, lags):
    start = max(lags)
    return np.concatenate([(raw[start-lag:len(raw)-lag]-mean)/sd for lag in lags], axis=1)


class PairedMLP:
    """One sampled current target per row; input mask is NOT the loss mask."""

    def __init__(self, channels=86, hidden=32, seed=10):
        self.d = channels
        rng = np.random.default_rng(seed)
        sizes = [8*channels, hidden, hidden, channels]
        self.p = {}
        for j, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.p["w"+str(j)] = rng.normal(0, np.sqrt(2/a), (a, b))
            self.p["b"+str(j)] = np.zeros(b)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    def inputs(self, x, targets, mode):
        mask = np.zeros_like(x) if mode == "joint" else np.ones_like(x)
        if mode == "self":
            mask[np.arange(len(x))[:, None], targets[:, None]+self.d*np.arange(1, 4)] = 0
        else:
            assert mode == "joint"
            mask[np.arange(len(x)), targets] = 1
        return np.concatenate((x*(1-mask), mask), axis=1)

    def forward(self, inputs):
        p = self.p
        h1 = np.maximum(inputs @ p["w0"]+p["b0"], 0)
        h2 = np.maximum(h1 @ p["w1"]+p["b1"], 0)
        return h2 @ p["w2"]+p["b2"], h1, h2

    def loss_grad(self, x, targets, mode):
        inputs = self.inputs(x, targets, mode)
        out, h1, h2 = self.forward(inputs)
        rows = np.arange(len(x))
        error = out[rows, targets]-x[rows, targets]
        delta = np.zeros_like(out)
        delta[rows, targets] = 2*error/len(x)
        grad = {"w2": h2.T @ delta, "b2": delta.sum(axis=0)}
        dh2 = (delta @ self.p["w2"].T)*(h2 > 0)
        grad.update(w1=h1.T @ dh2, b1=dh2.sum(axis=0))
        dh1 = (dh2 @ self.p["w1"].T)*(h1 > 0)
        grad.update(w0=inputs.T @ dh1, b0=dh1.sum(axis=0))
        assert np.isfinite(error).all() and all(np.isfinite(g).all() for g in grad.values())
        return float(np.mean(error**2)), grad

    def update(self, grad, cfg):
        self.step += 1
        b1, b2 = cfg["adam_beta1"], cfg["adam_beta2"]
        for k in self.p:
            self.m[k] = b1*self.m[k]+(1-b1)*grad[k]
            self.v[k] = b2*self.v[k]+(1-b2)*grad[k]**2
            self.p[k] -= cfg["learning_rate"]*(self.m[k]/(1-b1**self.step))/(np.sqrt(self.v[k]/(1-b2**self.step))+cfg["adam_epsilon"])
        assert all(np.isfinite(v).all() for v in self.p.values())

    def predict_all(self, x, mode, chunk=512):
        """Algebraically identical masked forward; share first-layer work."""
        p, d = self.p, self.d
        wv, wm = p["w0"][:4*d], p["w0"][4*d:]
        out = np.empty((len(x), d))
        for start in range(0, len(x), chunk):
            a = x[start:start+chunk]
            if mode == "joint":
                first = (a @ wv+p["b0"])[:, None, :]-a[:, :d, None]*wv[:d]+wm[:d]
            else:
                assert mode == "self"
                own = a.reshape(len(a), 4, d)[:, 1:].transpose(0, 2, 1)
                first = np.einsum("ncl,lch->nch", own, wv.reshape(4, d, -1)[1:], optimize=False)
                first += wm.sum(axis=0)-wm.reshape(4, d, -1)[1:].sum(axis=0)+p["b0"]
            h1 = np.maximum(first, 0)
            h2 = np.maximum(h1.reshape(-1, h1.shape[-1]) @ p["w1"]+p["b1"], 0).reshape(h1.shape)
            out[start:start+len(a)] = np.einsum("nch,hc->nc", h2, p["w2"], optimize=False)+p["b2"]
        assert np.isfinite(out).all()
        return out


def moments(arrays, floor):
    n = sum(len(x) for x in arrays)
    mu = sum(x.sum(axis=0) for x in arrays)/n
    sd = np.sqrt(sum(((x-mu)**2).sum(axis=0) for x in arrays)/n)
    return mu, np.maximum(sd, floor)


def tail_probability(values, sorted_cal):
    """Conservative >= ties and +1; caller includes search and persistence."""
    return (1+len(sorted_cal)-np.searchsorted(sorted_cal, values, side="left"))/(len(sorted_cal)+1)


def branch_pvalues(confirmed, calibration):
    return np.maximum.reduce([np.column_stack([tail_probability(confirmed[:, j], np.sort(c[:, j]))
                                              for j in range(confirmed.shape[1])]) for c in calibration])
