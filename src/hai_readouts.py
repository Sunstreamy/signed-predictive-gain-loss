import numpy as np


def finite(a):
    assert np.isfinite(a).all(), 'Nonfinite values'
    return a


def rolling(a, width=64):
    a = np.asarray(a, dtype=np.float64)
    c = np.concatenate([np.zeros((1,)+a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    return finite((c[width:]-c[:-width])/width)


def confirm(a):
    return np.minimum(np.minimum(a[:-2], a[1:-1]), a[2:])


def pair_score(loss, fit):
    c = (loss[:, :, :2]-fit['mean'])/fit['sd']-fit['center']
    q = np.einsum('nti,tij,ntj->nt', c, fit['precision'], c)
    finite(q)
    assert q.min() >= -1e-8
    return q, confirm(q.max(axis=1))


def tail(score, calibration):
    return np.maximum.reduce([(1+len(c)-np.searchsorted(np.sort(c), score, side='left'))/(len(c)+1)
                              for c in calibration])


def cutoffs(calibration, alphas):
    out = []
    for alpha in alphas:
        boundaries = []
        for c in calibration:
            # p(s)<=alpha iff s strictly exceeds this order statistic (including ties).
            k = int(np.floor(alpha*(len(c)+1)-1))
            assert k >= 0
            boundaries.append(float(np.sort(c)[len(c)-k-1]))
        out.append(dict(alpha=alpha, strict_upper_cutoff=max(boundaries), per_calibration_file=boundaries))
    return out


def gcad_split(length, history=5):
    cut = int(.8*length)
    # Every raw row is assigned to only one internal fit/validation role.
    return np.arange(history, cut), np.arange(cut+history, length), cut
