import numpy as np
import residual as residual_core
import model as old


def qualify_q3(cross, pooled, cfg):
    fit = cfg["roles"]["fit"]
    base = residual_core.qualify(cross, cfg)
    output = []
    for r in base:
        rows = [a for a in pooled if a["target"] == r["target"] and a["file"] in fit]
        assert len(rows) == 2 and {a["file"] for a in rows} == set(fit)
        assert all(a["stage"] == "pooled_model" for a in rows)
        checks = [a["relative_gain"] >= cfg["eligibility_relative_gain"] and
                  a["positive_block_fraction"] >= cfg["eligibility_positive_block_fraction"] for a in rows]
        output.append(dict(target=r["target"], channel=rows[0]["channel"], q5=r["eligible"],
            q3=bool(r["eligible"] and all(checks)),
            cross_min_relative_gain=r["min_relative_gain"],
            cross_min_positive_block_fraction=r["min_positive_block_fraction"],
            pooled_fit_min_relative_gain=min(a["relative_gain"] for a in rows),
            pooled_fit_min_positive_block_fraction=min(a["positive_block_fraction"] for a in rows)))
    return output


def aggregate(loss, mean, sd, count):
    z = (loss-mean)/sd
    assert np.isfinite(z).all() and np.all(sd > 0)
    raw = z.max(axis=1)
    return raw, old.confirmed_score(raw, count), z.argmax(axis=1)
