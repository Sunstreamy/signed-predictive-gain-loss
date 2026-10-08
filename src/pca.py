import numpy as np


def blocks(arrays, batch=8192):
    for x in arrays:
        for begin in range(0, len(x), batch):
            yield np.asarray(x[begin:begin+batch])


def fit_scores(arrays, rank, ridge=1e-4, sd_floor=1e-8):
    count = sum(len(x) for x in arrays)
    d = arrays[0].shape[1]
    mean = sum(x.sum(axis=0) for x in arrays)/count
    variance = sum(((x-mean)**2).sum(axis=0) for x in blocks(arrays))/count
    sd = np.sqrt(variance)
    sd = np.where(sd < sd_floor, 1., sd)
    center = sum(((x-mean)/sd).sum(axis=0) for x in blocks(arrays))/count
    covariance = np.zeros((d, d))
    direct_entries = np.zeros(d)
    for x in blocks(arrays):
        z = (x-mean)/sd-center
        covariance += np.einsum("ni,nj->ij", z, z, optimize=True)
        # Independent elementwise diagonal and adjacent-column products.
        direct_entries += (z*np.roll(z, 1, axis=1)).sum(axis=0)
    covariance = covariance/count + ridge*np.eye(d)
    for j in range(d):
        assert np.isclose(covariance[j, (j-1) % d], direct_entries[j]/count, atol=1e-10, rtol=1e-10)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    precision = np.einsum("ik,jk,k->ij", vectors, vectors, 1/eigenvalues)
    np.testing.assert_allclose(covariance @ precision, np.eye(d), atol=1e-7)
    fit = dict(mean=mean, sd=sd, center=center, covariance=covariance, eigenvalues=eigenvalues,
               vectors=vectors, precision=precision, components=vectors[:, -rank:].T,
               logdet=np.array(np.log(eigenvalues).sum()))
    assert all(np.isfinite(x).all() for x in fit.values())
    return fit
