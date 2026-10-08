"""Normal-side replay with frozen parameters, or the fixed fresh-fit procedure."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import time
import shutil
import resource
import numpy as np
import te_replication as t
args=None

def check_budget(cfg):
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=cfg['budget']['process_peak_memory_bytes']

def train_self(raw, ids, common, cfg, run):
    rng = np.random.default_rng(cfg['self']['seed'])
    cohorts, indices = zip(*[t.sampled_cohort(raw, group, common, rng, 4096) for group in ids])
    x = np.concatenate(cohorts)
    model = t.dev.PairedMLP(52, 32, 10)
    logs, shuffles, targets = [], [], []
    scfg = cfg['self']
    for epoch in range(scfg['epochs']):
        order = rng.permutation(len(x))
        target = rng.integers(0, 52, len(x))
        shuffles.append(order); targets.append(target)
        total = 0.
        for start in range(0, len(x), scfg['batch_size']):
            a = x[order[start:start+scfg['batch_size']]]
            loss, grad = model.loss_grad(a, target[start:start+len(a)], 'self')
            model.update(grad, scfg)
            total += loss*len(a)/len(x)
        logs.append(dict(epoch=epoch+1, loss=total))
        print('SELF', logs[-1], flush=True)
    t.csv_write(run/'training_losses.csv', logs)
    np.savez_compressed(run/'models/self.npz', **model.p)
    np.savez_compressed(run/'models/training_schedule.npz', indices=np.stack(indices),
                        shuffles=np.array(shuffles), targets=np.array(targets), samples=x)
    return model, cohorts

def target_gain_rows(means, positives, count, role, run_id, stage, columns, targets):
    rows = []
    for j, i in enumerate(targets):
        es, ej, d = means[j]
        rows.append(dict(role=role, run=run_id, stage=stage, target=int(i), channel=columns[i],
                         es=float(es), ej=float(ej), gain=float(-d), relative_gain=float(-d/max(es, 1e-8)),
                         positive_blocks=int(positives[j]), blocks=count,
                         positive_block_fraction=float(positives[j]/count)))
    return rows

def fit_normal(cfg, run):
    started = time.monotonic()
    parameters = getattr(args, "parameters", None)
    raw = t.normal_array('training')
    roles = cfg['roles']
    ids = [roles['fit_A_training_ids'], roles['fit_B_training_ids']]
    arrays = [raw[i-1] for i in ids[0]+ids[1]]
    if parameters:
        common=t.npz(parameters/'input_normalization.npz')
        model=t.dev.PairedMLP(52,32,10); model.p=t.npz(parameters/'self.npz')
        corrections=[t.npz(parameters/f'correction_{n}.npz') for n in ['A','B','pooled']]
        samples=sample_r=None
        for p in parameters.glob('*.npz'): shutil.copyfile(p,run/'models'/p.name)
    else:
        common = t.input_moments(arrays)
        np.savez_compressed(run/'models/input_normalization.npz', **common)
        model, samples = train_self(raw, ids, common, cfg, run)
        sample_r = [a[:, :52]-model.predict_all(a, 'self') for a in samples]
        rcfg = dict(cfg['correction'], chunk=4096)
        corrections = [t.residual.fit_correction(a, b, rcfg) for a, b in zip(samples, sample_r)]
        corrections.append(t.residual.fit_correction(np.concatenate(samples), np.concatenate(sample_r), rcfg))
        for name, correction in zip(['A', 'B', 'pooled'], corrections):
            np.savez_compressed(run/f'models/correction_{name}.npz', **correction)
    fit_rows, stages = [], {}
    for group, run_ids in enumerate(ids):
        name = ['A', 'B'][group]
        cross_means, pooled_means = [], []
        cross_good, pooled_good = np.zeros(52, int), np.zeros(52, int)
        blocks = 0
        for k, run_id in enumerate(run_ids):
            features = t.lag_run(raw[run_id-1], common)
            residual = features[:, :52]-model.predict_all(features, 'self')
            cross = t.residual.predict_correction(features, corrections[1-group])
            correction = t.residual.predict_correction(features, corrections[2])
            losses = [t.residual.paired_losses(residual, c) for c in [cross, correction]]
            saved = dict(residual=residual, correction=correction, cross_correction=cross, loss=losses[1])
            for stage, loss in zip(['cross', 'pooled'], losses):
                means, good, count, block_values = t.gain_statistics(loss)
                if stage == 'cross':
                    cross_means.append(means); cross_good += good
                else:
                    pooled_means.append(means); pooled_good += good
                saved[stage+'_blocks'] = block_values
                fit_rows += target_gain_rows(means, good, count, 'fit_'+name, run_id, stage,
                                             cfg['source']['columns'], list(range(52)))
            blocks += count
            if not parameters: np.savez_compressed(run/f'normal/fit_{run_id:03d}.npz', **saved)
            if (k+1) % 50 == 0:
                print('FIT_GAIN', name, k+1, flush=True)
            check_budget(cfg)
        stages['cross_'+name] = (np.mean(cross_means, axis=0), cross_good, blocks)
        stages['pooled_'+name] = (np.mean(pooled_means, axis=0), pooled_good, blocks)
    table = t.qualify(stages, cfg)
    targets = [r['target'] for r in table if r['qualified']]
    t.csv_write(run/'fit_run_gains.csv', fit_rows)
    t.csv_write(run/'qualification.csv', table)
    t.dump(run/'qualification.json', dict(utc=t.utc(), targets=targets,
           channels=[cfg['source']['columns'][i] for i in targets], table=table,
           no_calibration_audit_or_fault_selection=True, conditional_on_shared_self=True))
    if parameters:
        expected=json.loads((parameters/'qualification.json').read_text())['targets']
        assert targets==expected, 'Recomputed normal qualification differs; do not relax checks'
    print('QUALIFIED', len(targets), [cfg['source']['columns'][i] for i in targets], flush=True)
    if not targets:
        t.dump(run/'normal_completion.json',dict(status='empty_qualification_applicability_result',
               seconds=time.monotonic()-started, utc=t.utc(), fault_evaluation='not_executed',
               calibration='not_executed', matched_methods='N/A', no_rule_relaxation=True))
        return

    if parameters:
        readout=t.npz(parameters/'readout.npz'); pca=t.npz(parameters/'pca.npz')
    else:
        fit_losses=[t.npz(run/f'normal/fit_{i:03d}.npz')['loss'][:,targets] for i in ids[0]+ids[1]]
        readout=t.fit_readouts(fit_losses)
        np.savez_compressed(run/'models/readout.npz',**readout)
        pca=t.fit_pca(arrays,rank=34)
        np.savez_compressed(run/'models/pca.npz',**pca)
        del fit_losses
    del arrays,raw,samples,sample_r
    raw = t.normal_array('testing')
    gain_rows, cal = [], {}
    for role in ['cal_A', 'cal_B', 'audit_A', 'audit_B']:
        for k, run_id in enumerate(roles[role+'_testing_ids']):
            x = raw[run_id-1]
            features = t.lag_run(x, common)
            residual = features[:, :52]-model.predict_all(features, 'self')
            correction = t.residual.predict_correction(features, corrections[2])
            loss = t.residual.paired_losses(residual[:, targets], correction[:, targets])
            means, good, count, block_values = t.gain_statistics(loss)
            gain_rows += target_gain_rows(means,good,count,role,run_id,'pooled',cfg['source']['columns'],targets)
            scores, extra = t.score_readouts(loss, x, readout, pca)
            np.savez_compressed(run/f'normal/test_{run_id:03d}.npz', residual=residual[:, targets],
                                correction=correction[:, targets], loss=loss, blocks=block_values,
                                **scores,pca_point=extra['pca_point'])
            if role.startswith('cal'):
                for j, m in enumerate(t.METHODS): cal.setdefault(role+'_'+str(j), []).append(scores[m])
            if (k+1) % 50 == 0: print('NORMAL_SCORE', role, k+1, flush=True)
            check_budget(cfg)
    t.csv_write(run/'normal_run_gains.csv', gain_rows)
    cal = {k:np.concatenate(v) for k,v in cal.items()}
    np.savez_compressed(run/'models/calibration.npz', **cal)
    thresholds = {m:t.cutoffs([cal['cal_A_'+str(j)],cal['cal_B_'+str(j)]],t.ALPHAS)
                  for j,m in enumerate(t.METHODS)}
    t.dump(run/'thresholds.json',thresholds)
    metrics, events = [], []
    for role in ['cal_A','cal_B','audit_A','audit_B']:
        for run_id in roles[role+'_testing_ids']:
            saved=t.npz(run/f'normal/test_{run_id:03d}.npz')
            flags_to_save={}
            for j,m in enumerate(t.METHODS):
                ps=t.tail(saved[m],[cal['cal_A_'+str(j)],cal['cal_B_'+str(j)]])
                for c in thresholds[m]:
                    alpha=c['alpha']; flags=ps<=alpha
                    np.testing.assert_array_equal(flags,saved[m]>c['strict_upper_cutoff'])
                    flags_to_save[f'{j}_{alpha}']=flags
                    base=dict(role=role,run=run_id,method=m,alpha=alpha)
                    for support,start in [('native',t.FIRST[j]),('common',81)]:
                        metrics.append(dict(**base,support=support,**t.flag_metrics(flags,t.FIRST[j],start)))
                    for horizon,stop in [('full',960),('early',240)]:
                        events.append(dict(**base,horizon=horizon,**t.event_metrics(flags,t.FIRST[j],stop=stop)))
            np.savez_compressed(run/f'normal/test_{run_id:03d}.flags.npz',**flags_to_save)
    t.csv_write(run/'normal_run_metrics.csv',metrics)
    t.csv_write(run/'normal_pseudo_events.csv',events)
    t.dump(run/'normal_completion.json',dict(status='normal_side_complete',utc=t.utc(),
            seconds=time.monotonic()-started,qualified_targets=targets,normal_runs=1000,
            calibration_runs=100,audit_runs=400,fault_data_acquired=False,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def main():
    global args
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--parameters',type=Path)
    args=p.parse_args(); t.DATA=args.data_dir; run=t.new_output(args.output)
    for n in ['models','normal','fault']: (run/n).mkdir()
    cfg=json.loads(t.CONFIG.read_text()); fit_normal(cfg,run)
    t.execution_manifest(run,cfg,list((run/'models').glob('*.npz')))

if __name__=='__main__': main()
