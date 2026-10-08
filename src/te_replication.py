import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import model as dev
import residual
from windows import stratified_indices, rolling_mean
from pca import fit_scores as fit_pca
from hai_readouts import cutoffs, tail
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/te.json'
RUN = ROOT / 'outputs/te'
DATA = ROOT / 'data/te'
METHODS = ['S_QTE', 'Jr_QTE', 'Dr_QTE', 'PairError-QTE', 'PCA-SPE']
FIRST = [81, 81, 81, 81, 65]
ALPHAS = [.005, .01, .02]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def npz(path):
    with np.load(path) as data:
        return dict(data)


def finite(a):
    assert np.isfinite(a).all(), 'Nonfinite values'
    return a


def normal_array(split):
    length = 500 if split == 'training' else 960
    path = DATA/'decoded'/f'{split}.f64'
    a = np.memmap(path, dtype='<f8', mode='r', shape=(500, length, 55))
    assert np.array_equal(a[:, :, 0], np.zeros((500, length)))
    assert np.array_equal(a[:, :, 1], np.broadcast_to(np.arange(1, 501)[:, None], (500, length)))
    assert np.array_equal(a[:, :, 2], np.broadcast_to(np.arange(1, length+1), (500, length)))
    return finite(a[:, :, 3:])


def lag_run(raw, common):
    return finite(dev.lag_major(raw, common['mean'], common['sd'], [0, 1, 4, 16]))


def sampled_cohort(raw, ids, common, rng, count):
    n = raw.shape[1]-16
    ix = stratified_indices(len(ids)*n, count, rng)
    out = np.empty((count, 4*raw.shape[2]))
    for k, run_id in enumerate(ids):
        use = np.flatnonzero(ix//n == k)
        out[use] = lag_run(raw[run_id-1], common)[ix[use] % n]
    return finite(out), ix


def input_moments(arrays):
    n = sum(len(a) for a in arrays)
    mean = sum(a.sum(axis=0) for a in arrays)/n
    sd = np.sqrt(sum(((a-mean)**2).sum(axis=0) for a in arrays)/n)
    return dict(mean=finite(mean), sd=np.where(sd < 1e-8, 1., finite(sd)))


def confirm(values):
    return np.minimum(np.minimum(values[:-2], values[1:-1]), values[2:])


def gain_statistics(loss, block_size=20):
    means = finite(loss.mean(axis=0))
    blocks = np.stack([loss[i:i+block_size].mean(axis=0) for i in range(0, len(loss), block_size)])
    positives = (-blocks[:, :, 2] > 1e-10).sum(axis=0)
    return means, positives, len(blocks), finite(blocks)


def qualify(stages, cfg):
    """Stages are cross_A/cross_B/pooled_A/pooled_B cohort statistics."""
    q = cfg['qualification']
    rows = []
    for i, channel in enumerate(cfg['source']['columns']):
        row = dict(target=i, channel=channel)
        failures = []
        for name, (means, good, count) in stages.items():
            es, ej, delta = means[i]
            gain = -delta/max(es, 1e-8)
            fraction = good[i]/count
            row.update({name+'_es':float(es), name+'_gain':float(gain), name+'_positive_blocks':float(fraction)})
            if name.startswith('cross') and es < q['minimum_self_mse']:
                failures.append(name+':self_mse')
            if gain < q['minimum_relative_gain']:
                failures.append(name+':gain')
            if fraction < q['minimum_positive_block_fraction']:
                failures.append(name+':positive_blocks')
        row.update(qualified=not failures, failure_reasons=';'.join(failures))
        rows.append(row)
    return rows


def fit_readouts(losses):
    mean, sd = dev.moments(losses, 1e-8)
    u = np.concatenate([(x[:, :, :2]-mean[:, :2])/sd[:, :2] for x in losses])
    center = u.mean(axis=0)
    c = u-center
    cov = np.einsum('nti,ntj->tij', c, c)/len(c)
    reg = 1e-6*np.maximum(np.trace(cov, axis1=1, axis2=2)/2, 1e-8)
    cov += reg[:, None, None]*np.eye(2)
    return dict(mean=finite(mean), sd=finite(sd), pair_center=center, pair_covariance=cov,
                pair_precision=finite(np.linalg.inv(cov)))


def score_readouts(loss, raw, readout, pca):
    z = finite((loss-readout['mean'])/readout['sd'])
    pair = z[:, :, :2]-readout['pair_center']
    pair = finite(np.einsum('nti,tij,ntj->nt', pair, readout['pair_precision'], pair))
    assert pair.min() >= -1e-8
    scores = {m: confirm(z[:, :, k].max(axis=1)) for k, m in enumerate(METHODS[:3])}
    scores[METHODS[3]] = confirm(pair.max(axis=1))
    standard = (raw-pca['mean'])/pca['sd']-pca['center']
    remainder = standard-(standard @ pca['components'].T) @ pca['components']
    point = finite((remainder**2).sum(axis=1))
    scores[METHODS[4]] = confirm(rolling_mean(point, 64))
    return scores, dict(z=z, pair=pair, pca_point=point)


def flag_metrics(flags, first, start=None, stop=None):
    times = np.arange(first, first+len(flags))
    use = np.ones(len(flags), bool)
    if start is not None:
        use &= times >= start
    if stop is not None:
        use &= times < stop
    onsets = flags & ~np.r_[False, flags[:-1]]
    selected = flags & use
    edges = np.diff(np.r_[0, selected.astype(int), 0])
    lengths = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
    n = int(use.sum())
    return dict(points=n, alarm_points=int(selected.sum()), fpr=float(selected.sum()/n),
                alarm_episodes=int((onsets & use).sum()),
                episodes_per_hour=float((onsets & use).sum()/(n/20)),
                longest_alarm_minutes=int(max(lengths, default=0))*3, any_alarm=bool(selected.any()))


def event_metrics(flags, first, start=160, stop=960):
    onsets = flags & ~np.r_[False, flags[:-1]]
    times = np.arange(first, first+len(flags))
    event = (times >= start) & (times < stop)
    hit = times[event & onsets]
    delay = int(hit[0]-start) if len(hit) else None
    result = flag_metrics(flags, first, start, stop)
    result.update(detected=bool(len(hit)), delay_steps=delay,
                  delay_minutes=None if delay is None else 3*delay,
                  capped_delay_steps=stop-start if delay is None else delay,
                  any_hit=bool((event & flags).any()), pre_alarm=bool(flags[start-first-1]))
    return result


def four_sets(a, b):
    return dict(gained=a-b, lost=b-a, shared=a & b, neither=None)


def new_output(path):
    path = Path(path).resolve()
    if path == ROOT/'results' or ROOT/'results' in path.parents:
        raise ValueError('Archived results are read-only')
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Use a new output directory: '+str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_manifest(output, config, parameters):
    files = list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    dump(Path(output)/'execution.json', dict(
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in files},
        config=config, parameter_sha256={p.name:digest(p) for p in parameters},
        numpy=np.__version__, dtype='float64'))
