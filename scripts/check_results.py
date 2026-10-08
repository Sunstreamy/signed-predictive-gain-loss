"""Check archived result tables; this does not rerun a detector."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_statistics import ratio_ci


def read(dataset, name):
    with (ROOT / 'results' / dataset / name).open() as f:
        return list(csv.DictReader(f))


def true(row, key='detected'):
    return row[key] == 'True'


def equal(row, key, value):
    actual = row[key]
    if value is None:
        assert actual == '', (key, actual, value)
    else:
        assert np.isclose(float(actual), value, rtol=1e-12, atol=1e-12), (key, actual, value)


def hai():
    records = read('hai', 'event_results.csv')
    costs = read('hai', 'normal_background_metrics.csv')
    native = [r for r in records if r['view'] == 'native']
    identities = {(r['file'], r['event_id']) for r in native}
    assert len(identities) == 58
    groups = {'development': ['test4.csv'], 'confirmation_cohort_posthoc':
              ['test1.csv', 'test2.csv', 'test3.csv']}
    groups.update({f'test{i}.csv': [f'test{i}.csv'] for i in range(1, 5)})
    sets = {}
    summaries = read('hai', 'event_summary.csv')
    for s in summaries:
        names = groups[s['group']]
        rows = [r for r in native if r['file'] in names and r['method'] == s['method']
                and r['alpha'] == s['alpha']]
        hit = {(r['file'], r['event_id']) for r in rows if true(r)}
        sets[(s['group'], s['alpha'], s['method'])] = hit
        for k, v in [('events', len(rows)), ('detected', len(hit)),
                     ('missed', len(rows)-len(hit)), ('recall', len(hit)/len(rows))]:
            equal(s, k, v)
        delays = [int(r['delay']) for r in rows if true(r)]
        equal(s, 'detected_median_delay', float(np.median(delays)) if delays else None)
        equal(s, 'detected_max_delay', max(delays) if delays else None)
        equal(s, 'mean_capped_delay', np.mean([float(r['capped_delay']) for r in rows]))
        equal(s, 'mean_normalized_capped_delay',
              np.mean([float(r['normalized_capped_delay']) for r in rows]))
        clean = [r for r in costs if r['file'] in names and r['method'] == s['method']
                 and r['alpha'] == s['alpha'] and r['view'] == 'native' and r['selection'] == 'clean']
        equal(s, 'pooled_clean_fpr', sum(int(r['alarm_points']) for r in clean)/sum(int(r['points']) for r in clean))
        equal(s, 'macro_clean_fpr', np.mean([float(r['fpr']) for r in clean]))
        equal(s, 'worst_clean_fpr', max(float(r['fpr']) for r in clean))
        file_recall = [sum(true(r) for r in rows if r['file'] == f)/sum(r['file'] == f for r in rows) for f in names]
        equal(s, 'macro_recall', np.mean(file_recall))
    comparisons = read('hai', 'event_set_counts.csv')
    for s in comparisons:
        key = (s['group'], s['alpha'])
        d = sets[(*key, 'Dr_Q3')]
        ref = s['reference']
        if ref == 'reference_union_diagnostic_NOT_OR_system':
            b = set().union(*[v for k, v in sets.items() if k[:2] == key and k[2] != 'Dr_Q3'])
        else:
            b = sets[(*key, ref)]
        universe = {i for i in identities if i[0] in groups[s['group']]}
        for k, v in dict(gained=d-b, lost=b-d, shared=d & b, neither=universe-(d | b)).items():
            equal(s, k, len(v))
    for r in costs:
        equal(r, 'fpr', int(r['alarm_points'])/int(r['points']))
    for r in read('hai', 'file_leave_one_out.csv'):
        rows = [x for x in native if x['file'] in groups['confirmation_cohort_posthoc']
                and x['file'] != r['omitted'] and x['method'] == r['method'] and x['alpha'] == r['alpha']]
        equal(r, 'events', len(rows)); equal(r, 'detected', sum(true(x) for x in rows))
        equal(r, 'recall', sum(true(x) for x in rows)/len(rows))
    main = next(s for s in summaries if s['group'] == 'confirmation_cohort_posthoc'
                and s['method'] == 'Dr_Q3' and s['alpha'] == '0.01')
    assert int(main['detected']) == 4 and int(main['events']) == 34
    return dict(official_events=58, event_rows=len(records), summaries=len(summaries),
                comparison_rows=len(comparisons), normal_cost_rows=len(costs), Dr_holdout='4/34')


def te():
    records = read('te', 'fault_events.csv')
    identities = {(int(r['fault']), int(r['run'])) for r in records}
    assert len(identities) == 1000 and {i[0] for i in identities} == set(range(1, 21))
    assert len({i[1] for i in identities}) == 50
    clusters = {r['family']: r['cluster'] for r in json.loads((ROOT/'data_sources/te_family_clusters.json').read_text())}
    index = {(int(r['fault']), int(r['run']), r['method'], r['alpha'], r['horizon']): r for r in records}
    def select(group, method, alpha, horizon):
        rows = [r for r in records if r['method'] == method and r['alpha'] == alpha and r['horizon'] == horizon]
        if group.startswith('fault_'): rows = [r for r in rows if int(r['fault']) == int(group[6:])]
        elif group.startswith('panel_'): rows = [r for r in rows if r['panel'] == group[-1]]
        else: assert group == 'all'
        return rows
    summaries = read('te', 'fault_summary.csv')
    for s in summaries:
        rows = select(s['group'], s['method'], s['alpha'], s['horizon'])
        hits = [true(r) for r in rows]
        n = len(rows)
        delays = [int(r['delay_minutes']) for r in rows if true(r)]
        lo, hi, units = ratio_ci(rows, hits, [1]*n, clusters)
        fields = dict(runs=n, detected=sum(hits), missed=n-sum(hits), recall=sum(hits)/n,
                      recall_low=lo, recall_high=hi, family_units=units,
                      median_detected_delay_minutes=float(np.median(delays)) if delays else None,
                      p90_detected_delay_minutes=float(np.quantile(delays, .9)) if delays else None,
                      mean_capped_delay_minutes=3*np.mean([int(r['capped_delay_steps']) for r in rows]),
                      prealarm_fraction=sum(true(r, 'pre_alarm') for r in rows)/n,
                      any_hit_fraction=sum(true(r, 'any_hit') for r in rows)/n,
                      fault_alarm_fraction=sum(int(r['alarm_points']) for r in rows)/sum(int(r['points']) for r in rows),
                      alarm_episodes=sum(int(r['alarm_episodes']) for r in rows),
                      maximum_alarm_minutes=max(int(r['longest_alarm_minutes']) for r in rows))
        for k, v in fields.items(): equal(s, k, v)
    comparisons = read('te', 'event_sets_summary.csv')
    for s in comparisons:
        rows = select(s['group'], 'Dr_QTE', s['alpha'], s['horizon'])
        a = {(int(r['fault']), int(r['run'])) for r in rows if true(r)}
        b = {(int(r['fault']), int(r['run'])) for r in rows
             if true(index[(int(r['fault']), int(r['run']), s['reference'], s['alpha'], s['horizon'])])}
        universe = {(int(r['fault']), int(r['run'])) for r in rows}
        for k, v in dict(gained=a-b, lost=b-a, shared=a & b, neither=universe-(a | b)).items():
            equal(s, k, len(v))
        delta = [int(true(r))-int(true(index[(int(r['fault']), int(r['run']), s['reference'], s['alpha'], s['horizon'])])) for r in rows]
        lo, hi, units = ratio_ci(rows, delta, [1]*len(rows), clusters)
        for k, v in dict(net_recall_difference=np.mean(delta), difference_low=lo, difference_high=hi, family_units=units).items():
            equal(s, k, v)
    normal = read('te', 'normal_run_metrics.csv')
    for r in normal: equal(r, 'fpr', int(r['alarm_points'])/int(r['points']))
    qualification = read('te', 'qualification.csv')
    assert len(qualification) == 52 and sum(true(r, 'qualified') for r in qualification) == 40
    for horizon, expected in [('full',948),('early',854)]:
        s = next(s for s in summaries if s['group'] == 'all' and s['method'] == 'Dr_QTE'
                 and s['alpha'] == '0.01' and s['horizon'] == horizon)
        assert int(s['detected']) == expected
    return dict(fault_runs=1000, families=50, fault_types=20, event_rows=len(records),
                summaries=len(summaries), comparison_rows=len(comparisons),
                normal_cost_rows=len(normal), qualified_targets='40/52', Dr_full=948, Dr_early=854)


if __name__ == '__main__':
    print(json.dumps(dict(hai=hai(), te=te(), status='archived_tables_consistent'), indent=2))
