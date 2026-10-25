"""Merge check_clips.py output (data/check_clips.log) into data/clips_<source>.csv and set the final usable flag:
host's voice, cut length ok, and the cue clip re-transcribes as just "okay"/"all right". usage: clip_text.py <source>"""
import csv, os, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
# batch runs set SGU_TAG so each batch writes its own report/CSV files
SFX = f"_{os.environ['SGU_TAG']}" if os.environ.get('SGU_TAG') else ''
path = D / f'clips_{sys.argv[1]}{SFX}.csv'
if not path.exists(): sys.exit(0)
rows = list(csv.DictReader(open(path)))
txt = {}
for line in open(D / 'check_clips.log'):
    if '_cue.wav' in line:
        name, rest = line.split('_cue.wav', 1); txt[name.strip()] = rest.split('|', 1)[1].strip()
for x in rows:
    t = txt.get(f"{x['episode']}_h{x['handoff']}", '')
    x['clip_text'] = t
    x['clip_clean'] = int(bool(re.fullmatch(r"(okay|ok|all right|alright|alrighty|all righty)[.?!,]?", t.lower())))
    x['usable'] = int(x['host_voice'] == '1' and x['cut_ok'] == '1' and x['clip_clean'] == 1)
with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(sum(x['usable'] for x in rows), 'usable of', len(rows))
