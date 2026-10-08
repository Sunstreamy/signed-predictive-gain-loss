from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import hashlib
import urllib.request

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def check(entries, directory, download=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in entries:
        path=directory/e['name']
        if not path.exists() and download:
            partial=path.with_suffix(path.suffix+'.partial')
            if partial.exists(): raise FileExistsError(partial)
            url=e.get('url') or e.get('public_url')
            if not url: raise ValueError('Release URL is not available yet')
            with urllib.request.urlopen(url,timeout=120) as src, partial.open('wb') as dst:
                total=0
                while block:=src.read(1<<20):
                    total+=len(block)
                    if total>e['bytes']: raise ValueError('Unexpected source size: '+e['name'])
                    dst.write(block)
            if partial.stat().st_size!=e['bytes'] or digest(partial)!=e['sha256']:
                raise ValueError('Source hash mismatch: '+e['name'])
            partial.rename(path)
        if path.stat().st_size!=e['bytes'] or digest(path)!=e['sha256']:
            raise ValueError('Source hash mismatch: '+e['name'])
        rows.append(dict(file=e['name'],bytes=e['bytes'],sha256=e['sha256'],verified=True))
        print('SOURCE_VERIFIED',e['name'],flush=True)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['hai','te','parameters'])
    p.add_argument('--raw-dir',type=Path,required=True); p.add_argument('--download',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args(); source=json.loads((ROOT/'data_sources'/('model_artifacts.json' if a.dataset=='parameters' else f'{a.dataset}.json')).read_text())
    rows=check(([dict(e,name=e['artifact']) for e in source] if a.dataset=='parameters' else source['files']),a.raw_dir,a.download)
    if a.report: a.report.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__': main()
