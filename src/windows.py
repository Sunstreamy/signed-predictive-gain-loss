import numpy as np


def rolling_mean(x, width):
    total = np.concatenate((np.zeros((1,)+x.shape[1:]), np.cumsum(x, axis=0)), axis=0)
    return (total[width:]-total[:-width])/width


def stratified_indices(n, count, rng):
    edges = np.linspace(0, n, count+1).astype(int)
    assert np.all(np.diff(edges) > 0)
    return np.array([rng.integers(a, b) for a, b in zip(edges[:-1], edges[1:])])


def confirmed_score(raw, count=3):
    return np.min(np.lib.stride_tricks.sliding_window_view(raw, count, axis=0), axis=-1)
