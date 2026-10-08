"""Normal-only Self, residual correction, qualification and matched readouts."""
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
import shutil
import model, residual, qualification, pca
from te_replication import input_moments, npz, csv_write, dump, new_output, execution_manifest
from windows import stratified_indices

def sampled_data(arrays, common, cfg):
    rng=np.random.default_rng(cfg['seed']); xs=[]; indices=[]
    for a in arrays:
        features=model.lag_major(a,common['mean'],common['sd'],cfg['lags'])
        ix=stratified_indices(len(features),cfg['fit_examples_per_file'],rng)
        xs.append(features[ix]); indices.append(ix)
    return np.concatenate(xs),indices,rng

def train(x,rng,cfg):
    net=model.PairedMLP(cfg['channels'],cfg['hidden'],cfg['seed'])
    for epoch in range(cfg['epochs']):
        order=rng.permutation(len(x)); targets=rng.integers(0,cfg['channels'],len(x))
        for start in range(0,len(x),cfg['batch_size']):
            a=x[order[start:start+cfg['batch_size']]]
            _,grad=net.loss_grad(a,targets[start:start+len(a)],'self'); net.update(grad,cfg)
    return net

def fit(cache,output,cfg,parameters=None):
    run=new_output(output); names=cfg['roles']['fit']
    arrays=[np.load(Path(cache)/(n+'.x.npy'),mmap_mode='r') for n in names]
    if parameters:
        common=npz(parameters/'input_normalization.npz')
        net=model.PairedMLP(cfg['channels'],cfg['hidden'],cfg['seed']); net.p=npz(parameters/'self_model.npz')
        corrections=[npz(parameters/f'correction_{n}.npz') for n in ['A','B']]
        correction=npz(parameters/'correction_pooled.npz')
        for p in parameters.glob('*.npz'): shutil.copyfile(p,run/p.name)
    else:
        common=input_moments(arrays); np.savez_compressed(run/'input_normalization.npz',**common)
        x,indices,rng=sampled_data(arrays,common,cfg); net=train(x,rng,cfg)
        np.savez_compressed(run/'self_model.npz',**net.p)
        xs=np.split(x,2); rs=[v[:,:cfg['channels']]-net.predict_all(v,'self') for v in xs]
        corrections=[residual.fit_correction(a,r,cfg) for a,r in zip(xs,rs)]
        correction=residual.fit_correction(x,np.concatenate(rs),cfg)
        np.savez_compressed(run/'correction_pooled.npz',**correction)
    cross=[]; pooled=[]; losses=[]
    for i,(name,raw) in enumerate(zip(names,arrays)):
        h=model.lag_major(raw,common['mean'],common['sd'],cfg['lags'])
        r=h[:,:cfg['channels']]-net.predict_all(h,'self')
        for stage,c in [('cross_file',corrections[1-i]),('pooled_model',correction)]:
            loss=residual.paired_losses(r,residual.predict_correction(h,c,cfg['chunk']),cfg['window'])
            rows,_=residual.gain_rows(loss,cfg,name,stage)
            for row in rows: row['channel']=json.loads((ROOT/'data_sources/hai.json').read_text())['columns'][row['target']+1]
            (cross if stage=='cross_file' else pooled).extend(rows)
            if stage=='pooled_model': losses.append(loss)
    table=qualification.qualify_q3(cross,pooled,cfg); csv_write(run/'qualification.csv',table)
    q=[r['target'] for r in table if r['q3']]; dump(run/'qualification.json',dict(targets=q))
    if parameters:
        expected=json.loads((parameters/'qualification.json').read_text())['targets']
        assert q==expected,'Normal qualification differs; do not relax criteria'
        fitted=npz(parameters/'readout.npz'); mean,sd=fitted['mean'],fitted['sd']
    else: mean,sd=model.moments(losses,cfg['scale_floor'])
    np.savez_compressed(run/'readout.npz',mean=mean,sd=sd)
    if q and not parameters:
        u=np.concatenate([(a[:,q,:2]-mean[q,:2])/sd[q,:2] for a in losses])
        center=u.mean(axis=0); v=u-center
        covariance=np.einsum('nti,ntj->tij',v,v)/len(v)
        covariance+=1e-6*np.maximum(np.trace(covariance,axis1=1,axis2=2)/2,1e-8)[:,None,None]*np.eye(2)
        np.savez_compressed(run/'pair.npz',mean=mean[q,:2],sd=sd[q,:2],center=center,
            covariance=covariance,precision=np.linalg.inv(covariance))
    if not parameters: np.savez_compressed(run/'pca.npz',**pca.fit_scores(arrays,rank=57))
    execution_manifest(run,cfg,list(run.glob('*.npz')))
    print('Q3',q,'frozen-parameter qualification' if parameters else 'fresh fitting',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--parameters',type=Path); a=p.parse_args()
    fit(a.cache,a.output,json.loads((ROOT/'configs/hai.json').read_text()),a.parameters)

if __name__=='__main__': main()
