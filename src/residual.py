import numpy as np
from scipy.linalg import solve
from windows import rolling_mean


def feature_indices(target, d):
    own = target+d*np.arange(1, 4)
    cross = np.flatnonzero(np.arange(4*d) % d != target)
    return own, cross


def fit_correction(x, residual, cfg):
    """Project cross features off own lags/intercept; then ridge-fit Self residual."""
    n, d = len(x), residual.shape[1]
    mean = x.mean(axis=0)
    centered = x-mean
    gram = centered.T @ centered/n
    rhs = centered.T @ residual/n
    weights, intercept = np.zeros((4*d, d)), np.zeros(d)
    projection, betas, scales, ranks = [], [], [], []
    for i in range(d):
        own, cross = feature_indices(i, d)
        ghh = gram[np.ix_(own, own)]
        ghc = gram[np.ix_(own, cross)]
        eig, vec = np.linalg.eigh(ghh)
        keep = eig > max(cfg["feature_variance_floor"], cfg["projection_rcond"]*max(0., eig[-1]))
        inverse = (vec[:, keep]/eig[keep]) @ vec[:, keep].T
        a = inverse @ ghc
        covariance = gram[np.ix_(cross, cross)]-ghc.T @ a
        covariance = (covariance+covariance.T)/2
        variance = np.maximum(np.diag(covariance), 0.)
        active = variance > cfg["feature_variance_floor"]
        scale = np.where(active, np.sqrt(variance), 1.)
        normalized = covariance/scale[:, None]/scale[None, :]
        normalized[~active, :] = 0
        normalized[:, ~active] = 0
        cov_residual = (rhs[cross, i]-a.T @ rhs[own, i])/scale
        cov_residual[~active] = 0
        beta_scaled = solve(normalized+cfg["ridge_lambda"]*np.eye(len(cross)), cov_residual,
                            assume_a="pos", check_finite=True)
        beta = beta_scaled/scale
        weights[cross, i] = beta
        weights[own, i] = -a @ beta
        intercept[i] = -mean @ weights[:, i]
        assert weights[i, i] == 0, "Current target must remain invisible"
        projection.append(a)
        betas.append(beta)
        scales.append(scale)
        ranks.append(int(keep.sum()))
    model = dict(weights=weights, intercept=intercept, mean=mean,
                 projection=np.array(projection), cross_beta=np.array(betas),
                 cross_scale=np.array(scales), own_rank=np.array(ranks))
    assert all(np.isfinite(v).all() for v in model.values())
    correction = predict_correction(x, model, cfg["chunk"])
    before, after = np.mean(residual**2, axis=0), np.mean((residual-correction)**2, axis=0)
    penalty = cfg["ridge_lambda"]*np.sum((model["cross_beta"]*model["cross_scale"])**2, axis=1)
    np.testing.assert_allclose(before-after, np.mean(correction**2, axis=0)+2*penalty, atol=2e-7, rtol=2e-6)
    assert np.all(after <= before+2e-7)
    return model


def predict_correction(x, model, chunk=4096):
    out = np.empty((len(x), model["weights"].shape[1]))
    for start in range(0, len(x), chunk):
        a = x[start:start+chunk]
        out[start:start+len(a)] = a @ model["weights"]+model["intercept"]
    assert np.isfinite(out).all()
    return out


def paired_losses(residual, correction, width=64):
    es = rolling_mean(residual**2, width)
    ej = rolling_mean((residual-correction)**2, width)
    # The same frozen Self appears in both errors: D = c^2 - 2*r*c.
    delta = rolling_mean(correction**2-2*residual*correction, width)
    assert np.all(np.abs(delta-(ej-es)) <= 2e-7+2e-6*np.maximum(es, ej))
    return np.stack((es, ej, delta), axis=2)


def gain_rows(loss, cfg, file, stage):
    summary, blocks = [], []
    size = cfg["gain_block_seconds"]
    for i in range(loss.shape[1]):
        mean = loss[:, i, :].mean(axis=0)
        good = []
        for start in range(0, len(loss), size):
            value = loss[start:start+size, i].mean(axis=0)
            positive = bool(-value[2] > cfg["gain_numerical_tolerance"])
            good.append(positive)
            blocks.append(dict(stage=stage, file=file, target=i, start=start+79,
                points=min(size, len(loss)-start), es=float(value[0]), ej=float(value[1]),
                gain=float(-value[2]), relative_gain=float(-value[2]/max(value[0], cfg["scale_floor"])),
                positive=positive))
        summary.append(dict(stage=stage, file=file, target=i, es=float(mean[0]), ej=float(mean[1]),
            gain=float(-mean[2]), relative_gain=float(-mean[2]/max(mean[0], cfg["scale_floor"])),
            positive_block_fraction=float(np.mean(good)), blocks=len(good)))
    return summary, blocks


def qualify(cross_file_rows, cfg):
    assert set(r["file"] for r in cross_file_rows) == set(cfg["roles"]["fit"])
    result = []
    for i in range(cfg["channels"]):
        rows = [r for r in cross_file_rows if r["target"] == i]
        assert len(rows) == 2 and all(r["stage"] == "cross_file" for r in rows)
        checks = [bool(r["es"] >= cfg["eligibility_min_self_mse"] and
                       r["relative_gain"] >= cfg["eligibility_relative_gain"] and
                       r["positive_block_fraction"] >= cfg["eligibility_positive_block_fraction"]) for r in rows]
        result.append(dict(target=i, eligible=all(checks), per_file_checks=checks,
                           min_relative_gain=min(r["relative_gain"] for r in rows),
                           min_positive_block_fraction=min(r["positive_block_fraction"] for r in rows)))
    return result
