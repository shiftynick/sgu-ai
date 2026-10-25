"""Combine two extraction runs into data/extract/consensus/<n>.json, keeping only episodes where they agree.
Also settles the fiction: the website's answer key when it has one (checked against the host's reveal),
otherwise the reveal both runs agree on. Writes data/consensus_report.csv. A third run in data/extract/<runA>_tiebreak/ breaks ties.
usage: consensus.py <runA> <runB> <episodes>"""
import csv, json, os, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
# batch runs set SGU_TAG so each batch writes its own report/CSV files
SFX = f"_{os.environ['SGU_TAG']}" if os.environ.get('SGU_TAG') else ''
a, b, nums = sys.argv[1], sys.argv[2], sys.argv[3:]
eps = json.loads((D / 'episodes.json').read_text())
(D / 'extract' / 'consensus').mkdir(parents=True, exist_ok=True)
rows = []
for n in nums:
    pa, pb = D / 'extract' / a / f'{n}.json', D / 'extract' / b / f'{n}.json'
    row = {'episode': n, 'date': eps[n]['date'], 'key_source': eps[n].get('key_source', 'site')}
    if not (pa.exists() and pb.exists()):
        rows.append({**row, 'status': 'not extracted'}); continue
    ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
    ta = [(t['rogue'], t['pick']) for t in ra['turns']]; tb = [(t['rogue'], t['pick']) for t in rb['turns']]
    pc = D / 'extract' / f'{a}_tiebreak' / f'{n}.json'
    if ta != tb and pc.exists():  # a third run settles it if it matches one of the two exactly
        rc = json.loads(pc.read_text()); tc = [(t['rogue'], t['pick']) for t in rc['turns']]
        if tc == ta: rb = rc
        elif tc == tb: ra = rc
        ta = [(t['rogue'], t['pick']) for t in ra['turns']]; tb = [(t['rogue'], t['pick']) for t in rb['turns']]
        row['tiebreak'] = 'used'
    site, rev_a, rev_b = eps[n]['fiction'], ra.get('fiction_revealed'), rb.get('fiction_revealed')
    row.update(turns_a=ta, turns_b=tb, site_fiction=site, revealed_a=rev_a, revealed_b=rev_b)
    if not ta: status = 'no guesses found'
    elif ta != tb: status = 'runs disagree on picks'
    elif site and rev_a == rev_b and rev_a not in (0, site): status = 'reveal contradicts site key'
    elif not site and (rev_a != rev_b or not rev_a): status = 'no agreed reveal'
    else: status = 'ok'
    fiction = site or (rev_a if rev_a == rev_b else None)
    row.update(status=status, fiction=fiction)
    if status == 'ok':
        (D / 'extract' / 'consensus' / f'{n}.json').write_text(json.dumps({'fiction': fiction, 'host': ra['host'], 'turns': ra['turns']}, indent=1))
    rows.append(row)
keys = ['episode', 'date', 'status', 'tiebreak', 'key_source', 'site_fiction', 'revealed_a', 'revealed_b', 'fiction', 'turns_a', 'turns_b']
with open(D / f'consensus_report{SFX}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r['status'] for r in rows))
for r in rows:
    if r['status'] != 'ok': print(r['episode'], r['status'], r.get('site_fiction'), r.get('revealed_a'), r.get('revealed_b'), r.get('turns_a'), r.get('turns_b') if r.get('turns_b') != r.get('turns_a') else '')
