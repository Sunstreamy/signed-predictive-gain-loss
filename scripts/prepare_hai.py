from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import csv
import numpy as np
from check_sources import check
from evaluation import segments
from te_replication import dump

def read_csv(path, expected_columns):
    with Path(path).open() as f: columns=next(csv.reader(f))
    if columns!=expected_columns: raise ValueError('HAI schema mismatch')
    values=np.loadtxt(path,delimiter=',',skiprows=1,usecols=range(1,88),dtype=np.float64,ndmin=2)
    if not np.isfinite(values).all() or not np.isin(values[:,-1],[0,1]).all():
        raise ValueError('Nonfinite channels or invalid labels')
    with Path(path).open() as f:
        reader=csv.reader(f); next(reader)
        time=np.array([row[0] for row in reader])
    if len(time)!=len(values): raise ValueError('Timestamp/value row mismatch')
    time=time.astype('datetime64[s]').astype(np.int64)
    if not np.all(np.diff(time)==1): raise ValueError('Non-contiguous timestamps')
    return np.ascontiguousarray(values[:,:-1],dtype='<f8'),values[:,-1].astype(np.uint8),time

def prepare(raw, output, source):
    check(source['files'],raw)
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for e in source['files']:
        x,y,time=read_csv(Path(raw)/e['name'],source['columns'])
        if len(x)!=e['rows'] or len(segments(y!=0))!=e['event_count']: raise ValueError('HAI row/event count')
        for key,v in [('x',x),('labels',y),('time',time)]: np.save(output/(e['name']+'.'+key+'.npy'),v)
        actual=segments(y!=0)
        reference=[(r['start'],r['end_exclusive']) for r in source['events'] if r['file']==e['name']]
        if actual!=reference: raise ValueError('Official interval mismatch')
        records.append(dict(file=e['name'],rows=len(x),channels=x.shape[1],events=len(actual)))
        print('PREPARED',e['name'],len(x),len(actual),flush=True)
    dump(output/'official_events.json',source['events']); dump(output/'schema.json',records)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    prepare(a.raw_dir,a.output,json.loads((ROOT/'data_sources/hai.json').read_text()))

if __name__=='__main__': main()
