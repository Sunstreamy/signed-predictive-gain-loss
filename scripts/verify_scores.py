from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import csv
import hashlib
import numpy as np
from te_replication import dump

def compare(actual,expected,keys,skip_union=False,methods=None):
    with Path(actual).open() as f: aa=list(csv.DictReader(f))
    with Path(expected).open() as f: ee=list(csv.DictReader(f))
    index={tuple(r[k] for k in keys):r for r in ee}
    expected_scope=[r for r in ee if (methods is None or r.get('method', methods[0]) in methods) and
        (methods is None or 'reference' not in r or r['reference'] in methods or
         (not skip_union and r['reference']=='reference_union_diagnostic_NOT_OR_system'))]
    expected_keys={tuple(r[k] for k in keys) for r in expected_scope}
    actual_keys={tuple(r[k] for k in keys) for r in aa if not (skip_union and r.get('reference')=='reference_union_diagnostic_NOT_OR_system')}
    differences=[]; checked=0; excluded=0
    if expected_keys-actual_keys:
        differences.append(dict(column='missing_rows',count=len(expected_keys-actual_keys)))
    for row in aa:
        if skip_union and row.get('reference')=='reference_union_diagnostic_NOT_OR_system':
            excluded+=1; continue
        identity=tuple(row[k] for k in keys)
        if identity not in index:
            differences.append(dict(key=identity,column='row',actual='extra',expected='absent')); continue
        other=index[identity]
        for k,v in row.items():
            if k not in other: continue
            checked+=1; w=other[k]
            same=v==w
            if not same:
                try: same=bool(np.isclose(float(v),float(w),rtol=1e-12,atol=1e-12))
                except ValueError: pass
            if not same: differences.append(dict(key=identity,column=k,actual=v,expected=w))
    return dict(actual_rows=len(aa),reference_rows=len(ee),checked_cells=checked,expected_rows_in_scope=len(expected_scope),
        differences=differences,excluded_partial_reference_union_rows=excluded,exact_or_1e_12=not differences)

def audit_windows(dataset,scores,cache=None,parameters=None):
    from te_replication import npz
    import model
    checks=[]
    if dataset=='te':
        for p in sorted((scores/'fault').glob('*.npz')):
            saved=npz(p); r,c,loss=saved['residual'],saved['correction'],saved['loss']
            for start in [0,len(loss)//2,len(loss)-1]:
                a,b=r[start:start+64],c[start:start+64]
                direct=np.stack([(a*a).mean(axis=0),((a-b)**2).mean(axis=0),(b*b-2*a*b).mean(axis=0)],axis=1)
                checks.append(dict(file=p.name,window=start,max_abs=float(abs(direct-loss[start]).max()),
                    passed=bool(np.allclose(direct,loss[start],rtol=1e-9,atol=1e-8))))
    else:
        if cache is None or parameters is None: raise ValueError('HAI window checks require cache and parameters')
        q=json.loads((parameters/'qualification.json').read_text())['targets']
        common=npz(parameters/'input_normalization.npz'); correction=npz(parameters/'correction_pooled.npz')
        net=model.PairedMLP(86,32,10);net.p=npz(parameters/'self_model.npz')
        for p in sorted(scores.glob('*.loss.npz')):
            name=p.name.removesuffix('.loss.npz')
            raw=np.load(cache/(name+'.x.npy'),mmap_mode='r'); loss=npz(p)['loss']
            for start in [0,len(loss)//2,len(loss)-1]:
                # Independent explicit masked forward and direct W64 summation.
                chunk=raw[start:start+80]
                h=model.lag_major(chunk,common['mean'],common['sd'],[0,1,4,16])
                prediction=[]
                for target in q:
                    inputs=net.inputs(h,np.full(64,target),'self')
                    prediction.append(net.forward(inputs)[0][:,target])
                r=h[:,q]-np.stack(prediction,axis=1)
                c=(h@correction['weights']+correction['intercept'])[:,q]
                direct=np.stack([(r*r).mean(axis=0),((r-c)**2).mean(axis=0),(c*c-2*r*c).mean(axis=0)],axis=1)
                checks.append(dict(file=name,window=start,max_abs=float(abs(direct-loss[start]).max()),
                    passed=bool(np.allclose(direct,loss[start],rtol=1e-9,atol=1e-8))))
    return dict(checks=checks,rtol=1e-9,atol=1e-8,passed=bool(checks and all(r['passed'] for r in checks)),
        exhaustive_windows=False,selection='first, middle, last window of every saved stream')

def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['hai','te'])
    p.add_argument('--scores',type=Path,required=True); p.add_argument('--report',type=Path,required=True)
    p.add_argument('--window-checks',action='store_true');p.add_argument('--cache',type=Path);p.add_argument('--parameters',type=Path)
    a=p.parse_args(); ref=ROOT/'results'/a.dataset
    specifications={
        'hai': [('qualification.csv',['target']),('normal_target_gains.csv',['stage','file','target']),('event_results.csv',['file','view','method','alpha','event_id']),
                ('normal_background_metrics.csv',['file','view','method','alpha','selection']),
                ('event_summary.csv',['group','method','alpha']),
                ('event_set_counts.csv',['group','reference','alpha']),
                ('file_leave_one_out.csv',['method','alpha','omitted']),
                ('common_support_cost_differences.csv',['file','reference','alpha'])],
        'te': [('qualification.csv',['target']),('normal_gain_summary.csv',['stage','role','target']),('normal_run_gains.csv',['role','run','stage','target']),('normal_run_metrics.csv',['role','run','method','alpha','support']),
               ('normal_pseudo_events.csv',['role','run','method','alpha','horizon']),
               ('normal_summary.csv',['group','method','alpha','support']),
               ('fault_events.csv',['fault','run','method','alpha','horizon']),
               ('fault_summary.csv',['group','method','alpha','horizon']),
               ('event_sets_summary.csv',['group','reference','alpha','horizon']),
               ('family_results.csv',['family','method','alpha','horizon'])]}
    completion=json.loads((a.scores/'completion.json').read_text()) if a.dataset=='hai' else None
    partial=bool(completion and len(completion['methods'])<11)
    methods=completion['methods'] if completion else ['S_QTE','Jr_QTE','Dr_QTE','PairError-QTE','PCA-SPE']
    rows={n:compare(a.scores/n,ref/n,k,skip_union=partial,methods=methods) for n,k in specifications[a.dataset] if (a.scores/n).exists()}
    expected_methods=11 if a.dataset=='hai' else 5
    summary='event_results.csv' if a.dataset=='hai' else 'fault_events.csv'
    with (a.scores/summary).open() as f: methods=sorted({r['method'] for r in csv.DictReader(f)})
    report=dict(dataset=a.dataset,methods=methods,all_methods_replayed=len(methods)==expected_methods,
        numerical_tolerance=dict(rtol=1e-12,atol=1e-12),tables=rows,
        passed=bool(rows and all(r['exact_or_1e_12'] for r in rows.values())),
        archived_strict_arithmetic_exceptions_remain=True)
    if a.window_checks:
        report['window_checks']=audit_windows(a.dataset,a.scores,a.cache,a.parameters)
        report['passed']=bool(report['passed'] and report['window_checks']['passed'])
    dump(a.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['tables','window_checks']},indent=2))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
