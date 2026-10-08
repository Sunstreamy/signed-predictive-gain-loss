"""Fixed normal-only GCAD training, all three seeds; no attack-based selection."""
from pathlib import Path
import argparse
import json
import random
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import gcad_adapter as g
import hai_readouts as readouts
from te_replication import dump, new_output

h=None

def log(run,event,**fields):
    print(event,fields,flush=True)

def scaled(name,scale,device):
    x=np.asarray((h.load(name)-scale['mean'])/scale['sd'],dtype=np.float32)
    readouts.finite(x)
    return torch.as_tensor(x,device=device)


def official_scaler(raw):
    fitted = StandardScaler().fit(raw)
    return dict(mean=fitted.mean_, sd=fitted.scale_)

def batches(data, indices, batch, history):
    offsets = torch.arange(-history, 0, device=data.device)
    for start in range(0, len(indices), batch):
        t = torch.as_tensor(indices[start:start+batch], device=data.device)
        yield data[t[:, None]+offsets], data[t]

def validation(model, data, indices, cfg):
    total, n = 0., 0
    with torch.no_grad():
        for name in h.FIT:
            for x, y in batches(data[name], indices[name], cfg['batch'], cfg['history']):
                loss = (model(x)[:, 0]-y).square().sum()
                total += float(loss.cpu()); n += y.numel()
    return total/n

def fit_gcad(seed, cfg, run, device):
    out = run/f'gcad_s{seed}'
    out.mkdir(exist_ok=True)
    assert not (out/'best.pt').exists(), 'Refuse silent retraining'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train, val, cuts = {}, {}, {}
    for name in h.FIT:
        train[name], val[name], cuts[name] = h.gcad_split(len(h.load(name)), cfg['history'])
    raw = np.concatenate([h.load(n)[:cuts[n]] for n in h.FIT])
    scale = official_scaler(raw)
    np.savez_compressed(out/'scaler.npz', **scale)
    del raw
    data = {n: scaled(n, scale, device) for n in h.FIT}
    model = g.build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    rng = np.random.default_rng(seed)
    # Same shuffled union of legal per-file windows; no cross-file windows.
    file_index = np.concatenate([np.full(len(train[n]), i) for i, n in enumerate(h.FIT)])
    target_index = np.concatenate([train[n] for n in h.FIT])
    best, patience, history = float('inf'), 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg['max_epochs']+1):
        model.train(); total, count = 0., 0
        order = rng.permutation(len(target_index))
        for start in range(0, len(order), cfg['batch']):
            selected = order[start:start+cfg['batch']]
            xs, ys = [], []
            for fi, name in enumerate(h.FIT):
                ix = target_index[selected[file_index[selected] == fi]]
                if len(ix):
                    x, y = next(batches(data[name], ix, len(ix), cfg['history']))
                    xs.append(x); ys.append(y)
            x, y = torch.cat(xs), torch.cat(ys)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)[:, 0]-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite GCAD training loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach().cpu())*len(y); count += len(y)
        model.eval()
        vl = validation(model, data, val, cfg)
        assert np.isfinite(vl)
        improved = vl < best
        if improved:
            best, patience = vl, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out/'best.pt')
        else:
            patience += 1
        row = dict(epoch=epoch, train_mse=total/count, validation_mse=vl, best=improved,
                   elapsed_seconds=time.perf_counter()-t0)
        history.append(row); h.dump(out/'training.json', history)
        log(run, 'epoch', seed=seed, **row)
        if patience >= cfg['patience']:
            break
    model.load_state_dict(torch.load(out/'best.pt', weights_only=True)); model.eval()
    h.dump(out/'training_split.json', {n: dict(rows=len(data[n]), cut=cuts[n], train_windows=len(train[n]),
        validation_windows=len(val[n]), train_last=int(train[n][-1]), validation_first=int(val[n][0])) for n in h.FIT})
    # First batch and seeded Bernoulli(0.2) sample of nonoverlapping train batches.
    ref_rng = random.Random(seed+10000)
    ref_sum = np.zeros((86, 86)); ref_n = 0; sampled = []
    for name in h.FIT:
        for bi, start in enumerate(range(0, len(train[name]), cfg['batch'])):
            use = ref_rng.random() <= cfg['reference_sample_p']
            if bi != 0 and not use:
                continue
            ix = train[name][start:start+cfg['batch']]
            x, y = next(batches(data[name], ix, len(ix), cfg['history']))
            _, _, a = g.graph_batch(model, x, y, cfg['sparse_threshold'])
            aa = h.finite(a.cpu().numpy()).astype(np.float64)
            ref_sum += aa.sum(axis=0); ref_n += len(aa)
            sampled.append(dict(file=name, first_target=int(ix[0]), last_target=int(ix[-1])))
    reference = ref_sum/ref_n
    # Fixed-model prediction moments: all legal windows in the two normal fit files.
    pred_windows = []
    for name in h.FIT:
        point = predict_errors(model, data[name], cfg)
        pred_windows.append(h.rolling(point))
    pw = np.concatenate(pred_windows)
    norm = dict(mean=pw.mean(axis=0), sd=np.maximum(pw.std(axis=0), 1e-8), reference=reference)
    np.savez_compressed(out/'normalization.npz', **norm)
    h.dump(out/'reference_sample.json', dict(windows=ref_n, seed=seed+10000, batches=sampled))
    # Check per-output autograd against the evaluation Jacobian on normal windows.
    checks = []
    for name in h.FIT:
        ix = train[name][[0, len(train[name])//2, -1]]
        x, y = next(batches(data[name], ix, len(ix), cfg['history']))
        pred, a = g.autograd_graph(model, x, y, cfg['sparse_threshold'])
        fast, _, b = g.graph_batch(model, x, y, cfg['sparse_threshold'])
        ds1 = g.dependency_score(a, torch.tensor(reference, dtype=torch.float32, device=device))
        ds2 = g.dependency_score(b, torch.tensor(reference, dtype=torch.float32, device=device))
        np.testing.assert_allclose(fast.cpu(), pred.cpu(), rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(ds2.cpu(), ds1.cpu(), rtol=3e-4, atol=3e-4)
        checks.append(dict(file=name, prediction_max_abs=float((pred-fast).abs().max().cpu()),
            graph_max_abs=float((a-b).abs().max().cpu()), score_max_abs=float((ds1-ds2).abs().max().cpu())))
    h.dump(out/'normal_autograd_check.json', checks)
    log(run, 'fit_complete', seed=seed, epochs=len(history), reference_windows=ref_n, seconds=time.perf_counter()-t0)
    return model, scale, norm

def predict_errors(model, data, cfg):
    out = []
    with torch.no_grad():
        for x, y in batches(data, np.arange(cfg['history'], len(data)), 1024, cfg['history']):
            out.append((model(x)[:, 0]-y).square().cpu().numpy())
    return h.finite(np.concatenate(out))

def main():
    global h
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=['cpu','mps','cuda'])
    a=p.parse_args(); full=json.loads((ROOT/'configs/gcad.json').read_text()); cfg=full['gcad']
    run=new_output(a.output); torch.set_num_threads(full['threads'])
    h=SimpleNamespace(FIT=['train1.csv','train3.csv'],gcad_split=readouts.gcad_split,
        load=lambda n:np.load(a.cache/(n+'.x.npy'),mmap_mode='r'),finite=readouts.finite,
        rolling=readouts.rolling,dump=dump)
    for seed in full['seeds']: fit_gcad(seed,cfg,run,a.device or full['device'])

if __name__=='__main__': main()
