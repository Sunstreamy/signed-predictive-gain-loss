from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from types import SimpleNamespace
import hai_readouts as readouts
from evaluation import segments,metrics,clean_mask,event_metrics
from te_replication import csv_write as write_csv,dump
from score_hai import METHODS
args=None; h=None

def role(name):
    if name=='test4.csv': return 'development'
    if name in ['test1.csv','test2.csv','test3.csv']: return 'confirmation_cohort_posthoc'
    return 'calibration' if name in ['train4.csv','train5.csv'] else 'seen_normal_audit'

def series(run,name,cfg):
    saved=np.load(run/(name+'.scores.npz'),allow_pickle=False)
    for method in json.loads((run/'completion.json').read_text())['methods']:
        first=70 if method.startswith('GCAD') else 65 if method=='PCA-SPE' else 81
        yield method,saved[method],first

def main():
    cfg = json.loads((ROOT/'configs/gcad.json').read_text()); run = args.scores
    completion = json.loads((run/'completion.json').read_text()); cfg['seeds'] = [10,11,12] if len(completion['methods']) == 11 else []
    official = json.loads((args.cache/'official_events.json').read_text())
    cal = {}
    for n in h.CAL:
        for method, score, first in series(run, n, cfg):
            cal.setdefault(method, []).append(score)
    normal, events, episodes, summaries, comparisons, cost_diffs = [], [], [], [], [], []
    decisions = {}
    for name in h.CAL+h.AUDIT+h.TEST:
        labels = h.load(name, 'labels')
        if name not in h.TEST: assert not labels.any()
        decisions[name] = {}
        for method, score, first in series(run, name, cfg):
            p = h.tail(score, cal[method]); h.finite(p)
            flags_all = np.array([p <= alpha for alpha in cfg['alphas']])
            out = run/'decisions'; out.mkdir(exist_ok=True)
            np.savez_compressed(out/(name+'.'+method+'.npz'), pvalues=p, alarms=flags_all, first=np.array(first))
            decisions[name][method] = flags_all[:, 81-first:]
            for ai, alpha in enumerate(cfg['alphas']):
                flags = flags_all[ai]; onsets = flags & ~np.r_[False, flags[:-1]]
                for view, offset, support in [('native', 0, first), ('common81', 81-first, 81)]:
                    f, o, t0 = flags[offset:], onsets[offset:], first+offset
                    selections = {'normal': np.ones(len(f), bool)} if name not in h.TEST else {
                        'background': labels[t0:] == 0, 'clean': clean_mask(labels, t0, support),
                        'full_stream': np.ones(len(f), bool)}
                    for selection, mask in selections.items():
                        normal.append(dict(file=name, role=role(name), view=view, first=t0, support=support,
                            method=method, alpha=alpha, selection=selection, **metrics(f, o, mask)))
                    for e in (e for e in official if e['file'] == name):
                        events.append(dict(file=name, role=role(name), view=view, method=method, alpha=alpha,
                            event_id=e['event_id'], start=e['start'], end_exclusive=e['end_exclusive'], length=e['length'],
                            **event_metrics(f, o, t0, e)))
                for a, b in segments(flags, first):
                    episodes.append(dict(file=name, role=role(name), method=method, alpha=alpha,
                        start=a, end_exclusive=b, seconds=b-a,
                        overlaps_attack=bool(np.asarray(labels[a:b]).any())))
    write_csv(run/'normal_background_metrics.csv', normal)
    write_csv(run/'event_results.csv', events)
    write_csv(run/'alarm_episodes.csv', episodes)
    methods = list(cal)
    groups = {'development': ['test4.csv'], 'confirmation_cohort_posthoc': h.TEST[:3]}
    groups.update({n: [n] for n in h.TEST})
    for alpha in cfg['alphas']:
        for group, names in groups.items():
            universe = {e['file']+'_'+e['event_id'] for e in official if e['file'] in names}
            sets = {}
            for method in methods:
                rows = [r for r in events if r['alpha'] == alpha and r['method'] == method and r['file'] in names and r['view'] == 'native']
                sets[method] = {r['file']+'_'+r['event_id'] for r in rows if r['detected']}
                delays = [r['delay'] for r in rows if r['detected']]
                per_file = [sum(r['detected'] for r in rows if r['file'] == n)/sum(r['file'] == n for r in rows) for n in names]
                costs = [r for r in normal if r['alpha'] == alpha and r['method'] == method and r['file'] in names
                         and r['view'] == 'native' and r['selection'] == 'clean']
                summaries.append(dict(group=group, method=method, alpha=alpha, events=len(rows),
                    detected=len(sets[method]), recall=len(sets[method])/len(rows), macro_recall=float(np.mean(per_file)),
                    missed=len(rows)-len(sets[method]), detected_median_delay=float(np.median(delays)) if delays else None,
                    detected_max_delay=max(delays) if delays else None, mean_capped_delay=float(np.mean([r['capped_delay'] for r in rows])),
                    mean_normalized_capped_delay=float(np.mean([r['normalized_capped_delay'] for r in rows])),
                    pooled_clean_fpr=sum(r['alarm_points'] for r in costs)/sum(r['points'] for r in costs),
                    macro_clean_fpr=float(np.mean([r['fpr'] for r in costs])), worst_clean_fpr=max(r['fpr'] for r in costs)))
            d = sets['Dr_Q3']; refs = [m for m in methods if m != 'Dr_Q3']
            refs_sets = {m: sets[m] for m in refs}
            if len(methods)==11:
                refs_sets['reference_union_diagnostic_NOT_OR_system'] = set().union(*refs_sets.values())
            for method, b in refs_sets.items():
                comparisons.append(dict(group=group, alpha=alpha, reference=method,
                    gained=sorted(d-b), lost=sorted(b-d), shared=sorted(d & b), neither=sorted(universe-(d | b))))
        for name in h.TEST+h.AUDIT:
            mask = clean_mask(h.load(name, 'labels'), 81, 81)
            for method in methods:
                if method == 'Dr_Q3': continue
                ai = cfg['alphas'].index(alpha)
                d, b = decisions[name]['Dr_Q3'][ai], decisions[name][method][ai]
                cost_diffs.append(dict(file=name, alpha=alpha, reference=method, common_clean_points=int(mask.sum()),
                    d_fpr=float(d[mask].mean()), b_fpr=float(b[mask].mean()),
                    d_only_alarm_seconds=int((d & ~b & mask).sum()), b_only_alarm_seconds=int((b & ~d & mask).sum()),
                    shared_alarm_seconds=int((d & b & mask).sum())))
    write_csv(run/'event_summary.csv', summaries)
    h.dump(run/'event_sets.json', comparisons)
    write_csv(run/'event_set_counts.csv', [dict(group=r['group'], alpha=r['alpha'], reference=r['reference'],
        **{k: len(r[k]) for k in ['gained', 'lost', 'shared', 'neither']}) for r in comparisons])
    write_csv(run/'common_support_cost_differences.csv', cost_diffs)
    leave = []
    for alpha in cfg['alphas']:
        for method in methods:
            for omitted in h.TEST[:3]:
                es = [r for r in events if r['view'] == 'native' and r['alpha'] == alpha and r['method'] == method
                      and r['file'] in h.TEST[:3] and r['file'] != omitted]
                leave.append(dict(method=method, alpha=alpha, omitted=omitted, events=len(es), detected=sum(r['detected'] for r in es),
                                  recall=sum(r['detected'] for r in es)/len(es)))
    write_csv(run/'file_leave_one_out.csv', leave)
    matrix = []
    for e in official:
        row = dict(file=e['file'], role=role(e['file']), event_id=e['event_id'], start=e['start'], length=e['length'])
        for method in methods:
            r = next(r for r in events if r['file'] == e['file'] and r['event_id'] == e['event_id'] and
                     r['method'] == method and r['alpha'] == .01 and r['view'] == 'native')
            row[method] = f"detected_delay_{r['delay']}s" if r['detected'] else ('pre_alarm_only' if r['any_hit'] else 'miss')
        matrix.append(row)
    write_csv(run/'official_58_event_matrix.csv', matrix)
    h.dump(run/'analysis_complete.json', dict(method_rows=methods, official_events=len(matrix), event_rows=len(events),
        background_rows=len(normal), episode_rows=len(episodes), decisions_recomputed_from_normal_calibration=True))
    print('All selected readouts, files and nominal levels evaluated.', flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--scores',type=Path,required=True)
    args=p.parse_args()
    h=SimpleNamespace(FIT=['train1.csv','train3.csv'],CAL=['train4.csv','train5.csv'],AUDIT=['train2.csv','train6.csv'],
        TEST=[f'test{i}.csv' for i in range(1,5)],METHODS=METHODS[:3],finite=readouts.finite,tail=readouts.tail,dump=dump,
        load=lambda n,kind='x':np.load(args.cache/(n+'.'+kind+'.npy'),mmap_mode='r'))
    main()
