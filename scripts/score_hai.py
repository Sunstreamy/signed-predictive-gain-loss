"""Replay frozen readouts or use explicitly refitted normal-only parameters."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import model, residual, qualification
import hai_readouts as h
from te_replication import npz, dump, csv_write, new_output, execution_manifest

METHODS=['S_Q3','Jr_Q3','Dr_Q3','PairError-Q3','PCA-SPE']
NAMES=[f'train{i}.csv' for i in range(1,7)]+[f'test{i}.csv' for i in range(1,5)]

def simple(raw,parameters,cfg):
    common=npz(parameters/'input_normalization.npz'); fit=npz(parameters/'readout.npz')
    q=json.loads((parameters/'qualification.json').read_text())['targets']
    if not q: raise ValueError('No qualified targets; no rule relaxation')
    net=model.PairedMLP(cfg['channels'],cfg['hidden'],cfg['seed']); net.p=npz(parameters/'self_model.npz')
    features=model.lag_major(raw,common['mean'],common['sd'],cfg['lags'])
    r=features[:,:cfg['channels']]-net.predict_all(features,'self',cfg['prediction_chunk'])
    c=residual.predict_correction(features,npz(parameters/'correction_pooled.npz'),cfg['chunk'])
    loss=residual.paired_losses(r[:,q],c[:,q],cfg['window'])
    _,matched,_=qualification.aggregate(loss,fit['mean'][q],fit['sd'][q],cfg['consecutive'])
    _,pair=h.pair_score(loss,npz(parameters/'pair.npz'))
    pca=npz(parameters/'pca.npz'); point=[]
    for start in range(0,len(raw),8192):
        z=(raw[start:start+8192]-pca['mean'])/pca['sd']-pca['center']
        rem=z-(z@pca['components'].T)@pca['components']
        point.append(np.einsum('ij,ij->i',rem,rem))
    spe=h.confirm(h.rolling(h.finite(np.concatenate(point))))
    return {**{m:matched[:,j] for j,m in enumerate(METHODS[:3])},METHODS[3]:pair,METHODS[4]:spe},loss

def gcad(raw,parameters,cfg,seed,device):
    import torch
    import gcad_adapter as g
    from train_gcad import batches
    opts=cfg['gcad']; directory=parameters/f'gcad_s{seed}'
    net=g.build_model(opts).to(device)
    net.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True)); net.eval()
    scale=npz(directory/'scaler.npz'); norm=npz(directory/'normalization.npz')
    data=torch.as_tensor(np.asarray((raw-scale['mean'])/scale['sd'],dtype=np.float32),device=device)
    reference=torch.tensor(norm['reference'],dtype=torch.float32,device=device)
    errors=[]; dep=[]
    for x,y in batches(data,np.arange(opts['history'],len(data)),opts['score_batch'],opts['history']):
        _,error,a=g.graph_batch(net,x,y,opts['sparse_threshold'])
        errors.append(h.finite(error.cpu().numpy())); dep.append(h.finite(g.dependency_score(a,reference).cpu().numpy()))
    pw=h.rolling(np.concatenate(errors))
    prediction=h.confirm(((pw-norm['mean'])/norm['sd']).max(axis=1))
    dependency=h.confirm(h.rolling(np.concatenate(dep)))
    return {f'GCAD-prediction_s{seed}':prediction,f'GCAD-dependency_s{seed}':dependency}

def score(cache,parameters,run,cfg,with_gcad=False,device='mps'):
    gcfg=json.loads((ROOT/'configs/gcad.json').read_text())
    if with_gcad:
        import torch
        torch.set_num_threads(gcfg['threads'])
    methods=METHODS+([f'GCAD-{kind}_s{s}' for s in gcfg['seeds'] for kind in ['prediction','dependency']] if with_gcad else [])
    files=[]; gains=[]
    q=json.loads((parameters/'qualification.json').read_text())['targets']
    columns=json.loads((ROOT/'data_sources/hai.json').read_text())['columns'][1:-1]
    for name in NAMES:
        raw=np.load(Path(cache)/(name+'.x.npy'),mmap_mode='r')
        values,loss=simple(raw,parameters,cfg)
        np.savez_compressed(run/(name+'.loss.npz'),loss=loss)
        if name.startswith('train'):
            loss=npz(run/(name+'.loss.npz'))['loss']
            rows,_=residual.gain_rows(loss,cfg,name,'pooled_model')
            for r in rows:
                target=q[r['target']]
                gains.append(dict(channel=columns[target],eligible=True,**dict(r,target=target)))
        if with_gcad and name not in cfg['roles']['fit']:
            for seed in gcfg['seeds']:
                print('GCAD_SCORE',name,seed,flush=True)
                values.update(gcad(raw,parameters,gcfg,seed,device))
        np.savez_compressed(run/(name+'.scores.npz'),**values)
        print('SCORED',name,flush=True)
        files.append(name)
    csv_write(run/'normal_target_gains.csv',gains)
    thresholds={}
    for m in methods:
        arrays=[npz(run/(n+'.scores.npz'))[m] for n in cfg['roles']['calibration']]
        thresholds[m]=h.cutoffs(arrays,cfg['alphas'])
    dump(run/'thresholds.json',thresholds)
    dump(run/'completion.json',dict(files=files,methods=methods,all_58_events_scored=True,parameters_refitted=False))
    execution_manifest(run,cfg,list(parameters.rglob('*.npz'))+list(parameters.rglob('*.pt')))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--with-gcad',action='store_true'); p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    a=p.parse_args(); run=new_output(a.output)
    score(a.cache,a.parameters,run,json.loads((ROOT/'configs/hai.json').read_text()),a.with_gcad,a.device)

if __name__=='__main__': main()
