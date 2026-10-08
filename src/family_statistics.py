import numpy as np


def ratio_ci(rows,numerators,denominators,clusters):
    """Carry all20 fault variants of a sampled family together."""
    keys=sorted({clusters[int(r['run'])] for r in rows})
    sums=np.array([sum(n for r,n in zip(rows,numerators) if clusters[int(r['run'])]==k) for k in keys],float)
    counts=np.array([sum(n for r,n in zip(rows,denominators) if clusters[int(r['run'])]==k) for k in keys],float)
    draws=np.random.default_rng(20260930).integers(0,len(keys),size=(999,len(keys)))
    values=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    lo,hi=np.quantile(values,[.025,.975])
    return float(lo),float(hi),len(keys)
