import numpy as np


def segments(flags, first=0):
    diff = np.diff(np.r_[False, flags, False].astype(int))
    return [(int(a+first), int(b+first)) for a, b in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1))]


def metrics(flags, onsets, selected):
    n = int(selected.sum()); count = int((flags & selected).sum())
    return dict(points=n, alarm_points=count, fpr=count/n if n else None,
                episodes=int((onsets & selected).sum()),
                episodes_per_hour=float((onsets & selected).sum()*3600/n) if n else None,
                longest_alarm_seconds=max((b-a for a, b in segments(flags & selected)), default=0))


def clean_mask(labels, first, support):
    c = np.r_[0, np.cumsum(labels != 0)]
    t = np.arange(first, len(labels))
    return c[t+1]-c[t-support] == 0


def event_metrics(flags, onsets, first, event):
    a, b = event['start']-first, event['end_exclusive']-first
    assert a >= 1 and b <= len(flags)
    hit = np.flatnonzero(onsets[a:b])
    delay = int(hit[0]) if len(hit) else None
    return dict(detected=bool(len(hit)), any_hit=bool(flags[a:b].any()), pre_alarm=bool(flags[a-1]),
        delay=delay, capped_delay=delay if delay is not None else b-a,
        normalized_capped_delay=(delay/(b-a)) if delay is not None else 1.,
        alarm_points=int(flags[a:b].sum()), alarm_fraction=float(flags[a:b].mean()),
        longest_alarm_seconds=max((e-s for s, e in segments(flags[a:b])), default=0),
        new_episodes=int(len(hit)))
