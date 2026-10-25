"""Download episode MP3s in parallel into data/audio/ (atomic: .part then rename). usage: download.py <episodes>"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
eps = json.loads((D / 'episodes.json').read_text())

def get(n):
    out = D / 'audio' / f'{n}.mp3'
    if out.exists(): return f'{n} cached'
    part = out.with_suffix('.part')
    r = subprocess.run(['curl', '-sL', '--retry', '3', '-m', '1200', '-o', str(part), eps[n]['mp3']])
    if r.returncode: return f'{n} FAILED curl {r.returncode}'
    part.rename(out); return f'{n} {out.stat().st_size / 1e6:.0f} MB'

with ThreadPoolExecutor(6) as ex:
    for msg in ex.map(get, sys.argv[1:]): print(msg, flush=True)
