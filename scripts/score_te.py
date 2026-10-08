"""All predetermined 20 x 50 faulty-testing runs; frozen normal parameters."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import csv
import time
import resource
import numpy as np
import te_replication as t

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--normal',type=Path,required=True); a=p.parse_args()
    cfg=json.loads(t.CONFIG.read_text()); run=a.normal
    assert json.loads((run/'normal_completion.json').read_text())['status']=='normal_side_complete'
    assert not list((run/'fault').glob('*.npz')), 'Use a new output directory'
    q=json.loads((run/'qualification.json').read_text())['targets']; assert q
    common=t.npz(run/'models/input_normalization.npz'); correction=t.npz(run/'models/correction_pooled.npz')
    model=t.dev.PairedMLP(52,32,10); model.p=t.npz(run/'models/self.npz')
    readout=t.npz(run/'models/readout.npz'); pca=t.npz(run/'models/pca.npz')
    cal=t.npz(run/'models/calibration.npz'); thresholds=json.loads((run/'thresholds.json').read_text())
    with (ROOT/'data_sources/te_selected_runs.csv').open() as f: selected=list(csv.DictReader(f))
    raw=np.memmap(a.data_dir/'external_replication_v1/selected.f64',dtype='<f8',mode='r',shape=(1000,960,55))
    for k,r in enumerate(selected):
        assert np.all(raw[k,:,0]==int(r['fault'])) and np.all(raw[k,:,1]==int(r['run']))
        assert np.array_equal(raw[k,:,2],np.arange(1,961))
    started=time.monotonic(); events=[]; metrics=[]
    for k,meta in enumerate(selected):
        f,rid=int(meta['fault']),int(meta['run']);x=raw[k,:,3:]
        features=t.lag_run(x,common)
        residual=features[:,:52]-model.predict_all(features,'self')
        c=t.residual.predict_correction(features,correction)
        loss=t.residual.paired_losses(residual[:,q],c[:,q])
        scores,extra=t.score_readouts(loss,x,readout,pca)
        decisions={};panel='A' if rid in cfg['roles']['fault_panel_A_testing_ids'] else 'B'
        for j,m in enumerate(t.METHODS):
            ps=t.tail(scores[m],[cal[f'cal_A_{j}'],cal[f'cal_B_{j}']])
            for entry in thresholds[m]:
                alpha=entry['alpha'];flag=ps<=alpha
                np.testing.assert_array_equal(flag,scores[m]>entry['strict_upper_cutoff'])
                decisions[f'{j}_{alpha}']=flag
                base=dict(fault=f,run=rid,panel=panel,method=m,alpha=alpha)
                for horizon,stop_at in [('full',960),('early',240)]:
                    events.append(dict(**base,horizon=horizon,**t.event_metrics(flag,t.FIRST[j],stop=stop_at)))
                for support,first in [('native',t.FIRST[j]),('common',81)]:
                    for region,stop_at in [('all',960),('pre_fault',160)]:
                        metrics.append(dict(**base,support=support,region=region,
                                            **t.flag_metrics(flag,t.FIRST[j],first,stop_at)))
        np.savez_compressed(run/f'fault/f{f:02d}_r{rid:03d}.npz',residual=residual[:,q],
                            correction=c[:,q],loss=loss,**scores,pca_point=extra['pca_point'],**decisions)
        if (k+1)%50==0:
            print('FAULT_SCORED',k+1,'of1000; no performance summaries yet',flush=True)
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=cfg['budget']['process_peak_memory_bytes']
    assert len(events)==30000
    t.csv_write(run/'fault_events.csv',events);t.csv_write(run/'fault_run_metrics.csv',metrics)
    t.dump(run/'fault_completion.json',dict(status='all1000_complete',utc=t.utc(),runs=1000,
          seconds=time.monotonic()-started,families=50,fault_types=20,
          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__': main()
