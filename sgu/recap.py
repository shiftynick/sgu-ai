"""Independent check on the picks: ask Haiku (Claude CLI) only for the host's recap during the reveal
("Bob and Jay think this one is the fiction") and write data/extract/recap/<n>.json (per item: who chose it, and whether that rules out everyone else).
Then flag consensus picks that contradict it in data/recap_check.csv. usage: recap.py <episodes>"""
import csv, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import roster
D = Path(__file__).resolve().parent.parent / 'data'
eps = json.loads((D / 'episodes.json').read_text())
# batch runs set SGU_TAG so each batch writes its own report/CSV files
SFX = f"_{os.environ['SGU_TAG']}" if os.environ.get('SGU_TAG') else ''

PROMPT = """Below is the "Science or Fiction" segment of The Skeptics' Guide to the Universe (automatic transcript, no speaker labels, names may be misspelled: "Kara" is Cara).
During the reveal, the host goes through the items and recaps who chose each one as the fiction, e.g. "Bob and Jay, you think this one is the fiction. Everyone else thinks this one is science."
Using ONLY those recap statements by the host (not the panelists' own turns), for each item the host recaps give:
- item: the item number
- chose: the panelists the host says chose it as the fiction (names from: {roster}); empty if he says nobody did ("you all think this one is science")
- complete: true if the host's words also mean nobody else chose it ("the rest of you think this one is science", "you all think this one is science", or he names everyone); false if he does not say
- evidence: the host's words, quoted
Skip items the host does not recap.

ITEMS
{items}

TRANSCRIPT
{text}"""

def schema(names):
    return {'type': 'object', 'additionalProperties': False, 'required': ['items'], 'properties': {'items': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['item', 'chose', 'complete', 'evidence'], 'properties': {
            'item': {'type': 'integer', 'enum': [1, 2, 3]}, 'chose': {'type': 'array', 'items': {'type': 'string', 'enum': names}},
            'complete': {'type': 'boolean'}, 'evidence': {'type': 'string'}}}}}}

def run(n):
    out = D / 'extract' / 'recap' / f'{n}.json'
    if out.exists() or not (D / 'seg' / f'{n}.json').exists(): return
    ep, seg = eps[n], json.loads((D / 'seg' / f'{n}.json').read_text())
    items = '\n'.join(f"Item #{it['num']}: {it['text'] or '(read it from the transcript)'}" for it in ep['items'])
    cmd = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--no-session-persistence', '--output-format', 'json',
           '--system-prompt', 'You extract structured data from podcast transcripts. Answer only with the requested JSON.',
           '--json-schema', json.dumps(schema(roster(ep)))]
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, input=PROMPT.format(roster=', '.join(roster(ep)), items=items, text=seg['text']),
                           capture_output=True, text=True, cwd=cwd, timeout=900)
    try:
        res = json.loads(r.stdout); data = res.get('structured_output') or json.loads(res['result'])
    except Exception:
        print(n, 'FAILED', (r.stderr or r.stdout)[-300:], flush=True); return
    out.write_text(json.dumps(data, indent=1)); print(n, [(x['item'], x['chose'], x['complete']) for x in data['items']], flush=True)

if __name__ == '__main__':
    (D / 'extract' / 'recap').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex: list(ex.map(run, sys.argv[1:]))
    rows = []
    for n in sys.argv[1:]:
        c, r = D / 'extract' / 'consensus' / f'{n}.json', D / 'extract' / 'recap' / f'{n}.json'
        if not (c.exists() and r.exists()): continue
        recap = json.loads(r.read_text())['items']
        for t in json.loads(c.read_text())['turns']:
            named = [x['item'] for x in recap if t['rogue'] in x['chose']]
            excluded = [x['item'] for x in recap if x['complete'] and t['rogue'] not in x['chose']]
            if named and t['pick'] in named and len(named) == 1: status = 'agrees'
            elif named or t['pick'] in excluded: status = 'CONTRADICTS'
            else: status = 'no recap'
            rows.append({'episode': n, 'rogue': t['rogue'], 'pick': t['pick'], 'recap_pick': named, 'excluded': excluded, 'status': status})
    with open(D / f'recap_check{SFX}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['episode', 'rogue', 'pick', 'recap_pick', 'excluded', 'status']); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter(r['status'] for r in rows))
    for r in rows:
        if r['status'] == 'CONTRADICTS': print(' ', r)
