"""Score each backend's extraction against data/reference.json and against each other.
usage: compare.py <backend...>"""
import json, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
ref = {k: v for k, v in json.loads((D / 'reference.json').read_text()).items() if not k.startswith('_')}
backends = sys.argv[1:]
got = {b: {n: json.loads(p.read_text()) for n in ref if (p := D / 'extract' / b / f'{n}.json').exists()} for b in backends}

def picks(out): return {t['rogue']: t['pick'] for t in out['turns']}
def order(out): return [t['rogue'] for t in out['turns']]

print(f"{'backend':12} episodes  rogues found  extra rogues  picks right  order right  episodes fully right")
for b in backends:
    eps = got[b]; found = extra = right = total = ordok = full = 0
    for n, out in eps.items():
        r = dict(ref[n]); g = picks(out)
        total += len(r); found += sum(x in g for x in r); extra += sum(x not in r for x in g)
        right += sum(g.get(x) == p for x, p in r.items())
        ordok += order(out) == [x for x, _ in ref[n]]
        full += g == r and order(out) == [x for x, _ in ref[n]]
    print(f"{b:12} {len(eps):8}  {found:5}/{total:<6}  {extra:12}  {right:5}/{total:<5}  {ordok:5}/{len(eps):<5}  {full:5}/{len(eps)}")
print('\nper-episode picks (reference first; x = wrong or missing)')
for n in ref:
    row = [f"{n:>5} ref " + ' '.join(f'{x.split()[0]}={p}' for x, p in ref[n])]
    for b in backends:
        if n in got[b]:
            g = picks(got[b][n]); r = dict(ref[n])
            bad = [x.split()[0] for x in r if g.get(x) != r[x]] + [f'+{x.split()[0]}' for x in g if x not in r]
            row.append(f"{b}: {'ok' if not bad else 'x ' + ','.join(bad)}")
    print(' | '.join(row))
if len(backends) > 1:
    print('\nall backends agree on every pick in', sum(all(n in got[b] for b in backends) and len({json.dumps(picks(got[b][n]), sort_keys=True) for b in backends}) == 1 for n in ref), 'of', len(ref), 'episodes')
