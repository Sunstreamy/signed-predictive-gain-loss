from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import csv
from collections import defaultdict
import time
import numpy as np
import te_replication as t
from family_statistics import ratio_ci

def rows(name):
    with (t.RUN/name).open() as f: return list(csv.DictReader(f))
load=rows

def yes(row,key='detected'): return row[key]=='True'

def normal_summary():
    cfg=json.loads(t.CONFIG.read_text())
    q=json.loads((t.RUN/'qualification.json').read_text())['targets']
    metrics=rows('normal_run_metrics.csv');events=rows('normal_pseudo_events.csv')
    event={(int(r['run']),r['method'],float(r['alpha']),r['horizon']):r for r in events}
    families=cfg['roles']['fault_panel_A_testing_ids']+cfg['roles']['fault_panel_B_testing_ids']
    summary=[];uncertainty=[]
    selections={k:cfg['roles'][k+'_testing_ids'] for k in ['cal_A','cal_B','audit_A','audit_B']}
    selections['audit_all']=selections['audit_A']+selections['audit_B'];selections['audit_matched50']=families
    for group,ids in selections.items():
        for m in t.METHODS:
            for alpha in t.ALPHAS:
                for support in ['native','common']:
                    records={int(r['run']):r for r in metrics if r['method']==m and float(r['alpha'])==alpha and r['support']==support and int(r['run']) in ids}
                    selected=[records[i] for i in ids];fprs=np.array([float(r['fpr']) for r in selected])
                    full=np.array([event[(i,m,alpha,'full')]['detected']=='True' for i in ids])
                    early=np.array([event[(i,m,alpha,'early')]['detected']=='True' for i in ids])
                    n=sum(int(r['points']) for r in selected); alarms=sum(int(r['alarm_points']) for r in selected)
                    summary.append(dict(group=group,method=m,alpha=alpha,support=support,runs=len(ids),
                        pooled_fpr=alarms/n,mean_fpr=float(fprs.mean()),median_fpr=float(np.median(fprs)),
                        p90_fpr=float(np.quantile(fprs,.9)),p95_fpr=float(np.quantile(fprs,.95)),max_fpr=float(fprs.max()),
                        fraction_fpr_gt_1pct=float((fprs>.01).mean()),fraction_fpr_gt_2pct=float((fprs>.02).mean()),
                        fraction_fpr_gt_5pct=float((fprs>.05).mean()),any_alarm_run_fraction=float((fprs>0).mean()),
                        episodes=sum(int(r['alarm_episodes']) for r in selected),
                        episodes_per_hour=sum(int(r['alarm_episodes']) for r in selected)/(n/20),
                        max_alarm_minutes=max(int(r['longest_alarm_minutes']) for r in selected),
                        full_pseudo_event=float(full.mean()),early_pseudo_event=float(early.mean())))
                    if group=='audit_all' and support=='native':
                        draws=np.random.default_rng(20260930).integers(0,len(ids),size=(999,len(ids)))
                        for name,values in [('mean_FPR',fprs),('full_pseudo_event',full),('early_pseudo_event',early)]:
                            lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
                            uncertainty.append(dict(group=group,method=m,alpha=alpha,metric=name,estimate=float(values.mean()),
                                low=float(lo),high=float(hi),units=len(ids),replicates=999,seed=20260930))
    t.csv_write(t.RUN/'normal_summary.csv',summary);t.csv_write(t.RUN/'normal_uncertainty.csv',uncertainty)
    gains=rows('fit_run_gains.csv')+rows('normal_run_gains.csv')
    buckets=defaultdict(list)
    for r in gains: buckets[(r['stage'],r['role'],int(r['target']))].append(r)
    for i in q:
        buckets[('pooled','audit_all',i)]=[r for r in gains if r['stage']=='pooled' and r['role'].startswith('audit') and int(r['target'])==i]
    gain_summary=[]
    for (stage,role,i),items in sorted(buckets.items()):
        g=np.array([float(r['relative_gain']) for r in items]);es=np.array([float(r['es']) for r in items])
        absolute=np.array([float(r['gain']) for r in items]);b=np.array([float(r['positive_block_fraction']) for r in items])
        gain_summary.append(dict(stage=stage,role=role,target=i,channel=cfg['source']['columns'][i],qualified=i in q,runs=len(items),
            pooled_relative_gain=float(absolute.mean()/max(es.mean(),1e-8)),mean_run_relative_gain=float(g.mean()),
            min_run_relative_gain=float(g.min()),p05_run_relative_gain=float(np.quantile(g,.05)),
            median_run_relative_gain=float(np.median(g)),positive_gain_run_fraction=float((g>0).mean()),
            original_threshold_run_fraction=float(((g>=.05)&(b>=.6)&(es>=1e-6)).mean())))
    t.csv_write(t.RUN/'normal_gain_summary.csv',gain_summary)


def fault_summary():
    started=time.monotonic();run=t.RUN;cfg=json.loads(t.CONFIG.read_text())
    assert json.loads((run/'fault_completion.json').read_text())['status']=='all1000_complete'
    events=load('fault_events.csv');normal=load('normal_pseudo_events.csv')
    normal_metrics=load('normal_run_metrics.csv')
    normal_lookup={(int(r['run']),r['method'],float(r['alpha']),r['horizon']):r for r in normal}
    norm_fpr={(int(r['run']),r['method'],float(r['alpha'])):float(r['fpr']) for r in normal_metrics if r['support']=='native'}
    clusters={r['family']:r['cluster'] for r in json.loads((ROOT/'data_sources/te_family_clusters.json').read_text())}
    groups={'all':lambda r:True,'panel_A':lambda r:r['panel']=='A','panel_B':lambda r:r['panel']=='B'}
    for f in range(1,21):groups[f'fault_{f:02d}']=lambda r,f=f:int(r['fault'])==f
    index={(int(r['fault']),int(r['run']),r['method'],float(r['alpha']),r['horizon']):r for r in events}
    families=cfg['roles']['fault_panel_A_testing_ids']+cfg['roles']['fault_panel_B_testing_ids']
    summary=[];paired=[];family_rows=[];sets=[];all_ids={};membership=[]
    for alpha in t.ALPHAS:
        for horizon in ['full','early']:
            base=[r for r in events if r['method']=='Dr_QTE' and float(r['alpha'])==alpha and r['horizon']==horizon]
            assert len(base)==1000
            for r in base:
                f,rid=int(r['fault']),int(r['run'])
                row=dict(fault=f,run=rid,panel=r['panel'],alpha=alpha,horizon=horizon)
                row.update({m:yes(index[(f,rid,m,alpha,horizon)]) for m in t.METHODS})
                for m in [x for x in t.METHODS if x!='Dr_QTE']:
                    a,b=row['Dr_QTE'],row[m]
                    row['Dr_vs_'+m]='shared' if a and b else 'gained' if a else 'lost' if b else 'neither'
                membership.append(row)
            for m in t.METHODS:
                for rid in families:
                    items=[index[(f,rid,m,alpha,horizon)] for f in range(1,21)]
                    pseudo=normal_lookup[(rid,m,alpha,horizon)]
                    family_rows.append(dict(family=rid,cluster=clusters[rid],panel=items[0]['panel'],method=m,
                        alpha=alpha,horizon=horizon,detected=sum(yes(r) for r in items),runs=20,
                        recall=sum(yes(r) for r in items)/20,normal_pseudo_event=yes(pseudo),
                        normal_fpr=norm_fpr[(rid,m,alpha)],
                        mean_capped_delay_minutes=3*np.mean([int(r['capped_delay_steps']) for r in items])))
                for group,accept in groups.items():
                    items=[index[(int(r['fault']),int(r['run']),m,alpha,horizon)] for r in base if accept(r)]
                    n=len(items);det=[yes(r) for r in items];delays=[int(r['delay_minutes']) for r in items if yes(r)]
                    lo,hi,units=ratio_ci(items,det,[1]*n,clusters)
                    summary.append(dict(group=group,method=m,alpha=alpha,horizon=horizon,runs=n,
                        detected=sum(det),missed=n-sum(det),recall=sum(det)/n,recall_low=lo,recall_high=hi,family_units=units,
                        median_detected_delay_minutes=float(np.median(delays)) if delays else None,
                        p90_detected_delay_minutes=float(np.quantile(delays,.9)) if delays else None,
                        mean_capped_delay_minutes=3*np.mean([int(r['capped_delay_steps']) for r in items]),
                        prealarm_fraction=sum(yes(r,'pre_alarm') for r in items)/n,
                        any_hit_fraction=sum(yes(r,'any_hit') for r in items)/n,
                        fault_alarm_fraction=sum(int(r['alarm_points']) for r in items)/sum(int(r['points']) for r in items),
                        alarm_episodes=sum(int(r['alarm_episodes']) for r in items),
                        maximum_alarm_minutes=max(int(r['longest_alarm_minutes']) for r in items)))
                    normal_det=[yes(normal_lookup[(int(r['run']),m,alpha,horizon)]) for r in items]
                    difference=[int(a)-int(b) for a,b in zip(det,normal_det)]
                    plo,phi,punits=ratio_ci(items,difference,[1]*n,clusters)
                    paired.append(dict(group=group,method=m,alpha=alpha,horizon=horizon,records=n,family_units=punits,
                        fault_only=sum(a and not b for a,b in zip(det,normal_det)),normal_only=sum(b and not a for a,b in zip(det,normal_det)),
                        both=sum(a and b for a,b in zip(det,normal_det)),neither=sum(not a and not b for a,b in zip(det,normal_det)),
                        fault_recall=sum(det)/n,normal_pseudo_event=sum(normal_det)/n,
                        paired_difference=float(np.mean(difference)),difference_low=plo,difference_high=phi))
            for m in [x for x in t.METHODS if x!='Dr_QTE']:
                for group,accept in groups.items():
                    items=[r for r in base if accept(r)];universe={(int(r['fault']),int(r['run'])) for r in items}
                    a={(int(r['fault']),int(r['run'])) for r in items if yes(r)}
                    b={(f,rid) for f,rid in universe if yes(index[(f,rid,m,alpha,horizon)])}
                    parts=dict(gained=a-b,lost=b-a,shared=a & b,neither=universe-(a|b))
                    assert sum(map(len,parts.values()))==len(universe)
                    delta=[int(yes(r))-int(yes(index[(int(r['fault']),int(r['run']),m,alpha,horizon)])) for r in items]
                    lo,hi,units=ratio_ci(items,delta,[1]*len(items),clusters)
                    # Normal pseudo-event contrast over the same unique families, not repeated20 times.
                    selected_families=sorted({int(r['run']) for r in items})
                    na={i for i in selected_families if yes(normal_lookup[(i,'Dr_QTE',alpha,horizon)])}
                    nb={i for i in selected_families if yes(normal_lookup[(i,m,alpha,horizon)])}
                    sets.append(dict(group=group,reference=m,alpha=alpha,horizon=horizon,records=len(items),family_units=units,
                        **{k:len(v) for k,v in parts.items()},net_recall_difference=float(np.mean(delta)),difference_low=lo,difference_high=hi,
                        normal_gained=len(na-nb),normal_lost=len(nb-na),normal_shared=len(na & nb),normal_neither=len(set(selected_families)-(na|nb))))
                    if group=='all':all_ids[f'{alpha}/{horizon}/{m}']={k:sorted(v) for k,v in parts.items()}
    for name,items in [('fault_summary.csv',summary),('fault_normal_pairs.csv',paired),('family_results.csv',family_rows),
                       ('event_sets_summary.csv',sets),('event_membership.csv',membership)]:t.csv_write(run/name,items)
    t.dump(run/'event_sets.json',all_ids)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True); a=p.parse_args()
    t.RUN=a.scores; normal_summary()
    if (t.RUN/'fault_completion.json').exists(): fault_summary()
    print('Normal/fault summaries and family intervals recomputed.')
