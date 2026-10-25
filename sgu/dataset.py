"""Assemble data/dataset.csv: one row per usable handoff clip, with its label and the checks it passed.
Usable = host's voice, clean cut, clip re-transcribes as just the cue, and no recap contradiction in the episode.
usage: dataset.py"""
import csv, glob, json
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
eps = json.loads((D / 'episodes.json').read_text())
recap = [r for p in sorted(glob.glob(str(D / 'recap_check*.csv'))) for r in csv.DictReader(open(p))]
contradicted = {r['episode'] for r in recap if r['status'] == 'CONTRADICTS'}
recap_of = {(r['episode'], r['rogue']): r['status'] for r in recap}
rows = []
for r in (r for p in sorted(glob.glob(str(D / 'clips_consensus*.csv'))) for r in csv.DictReader(open(p))):
    if r['usable'] != '1' or r['episode'] in contradicted: continue
    rows.append({'episode': r['episode'], 'date': eps[r['episode']]['date'], 'handoff': r['handoff'], 'position': r['position'],
                 'prev_rogue': r['prev_rogue'], 'prev_pick': r['prev_pick'], 'fiction': r['fiction'], 'label_prev_correct': r['prev_correct'],
                 'next_rogue': r['next_rogue'], 'cue_text': r['clip_text'], 'cue_start': r['cue_start'], 'cue_end': r['cue_end'],
                 'clip': f"clips/{r['episode']}_h{r['handoff']}_cue.wav", 'clip_with_name': f"clips/{r['episode']}_h{r['handoff']}_cue_name.wav",
                 'host_sim': r['host_sim'], 'recap_check': recap_of.get((r['episode'], r['prev_rogue']), 'no recap'),
                 'key_source': eps[r['episode']].get('key_source', 'site')})
with open(D / 'dataset.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
pos = sum(r['label_prev_correct'] == '1' for r in rows)
print(f"{len(rows)} clips from {len({r['episode'] for r in rows})} episodes: {pos} after a correct guess, {len(rows) - pos} after a wrong one")
print('dropped episodes (recap contradiction):', sorted(contradicted, key=int))
print('labels whose pick the recap confirms directly:', sum(r['recap_check'] == 'agrees' for r in rows))
