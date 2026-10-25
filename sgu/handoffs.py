"""Find the host's "Okay/All right, <next rogue>" handoffs and label whether the previous rogue's pick was right.
Picks come from data/reference.json or from an extraction backend. Writes data/handoffs_<source>.csv.
A handoff needs a cue word right before the name; turns without one are reported as 'not found'.
usage: handoffs.py <reference|qwen|haiku|...> <episode numbers...>"""
import csv, json, os, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
# batch runs set SGU_TAG so each batch writes its own report/CSV files
SFX = f"_{os.environ['SGU_TAG']}" if os.environ.get('SGU_TAG') else ''
ALIASES = {'Bob': ['bob', 'robert'], 'Jay': ['jay', 'jeff'], 'Evan': ['evan', 'ivan'], 'Cara': ['cara', 'kara', 'kira', 'care', 'carrie'],
           'Rebecca': ['rebecca', 'becca'], 'Perry': ['perry']}
CUE = re.compile(r"^(okay|ok|all|alright|alrighty)$")

def tok(w): return re.sub(r"[^a-z']", '', w.lower())

def aliases(name): return ALIASES.get(name, [name.split()[0].lower()])

def turns_for(source, n):
    if source == 'reference':
        ref = json.loads((D / 'reference.json').read_text())[n]
        return [{'rogue': r, 'pick': p} for r, p in ref]
    return json.loads((D / 'extract' / source / f'{n}.json').read_text())['turns']

def cue_before(words, i, lo):
    """Index of an okay/ok/alright/"all right" cue within the 3 words before word i (allowing "and", "so"...), else None."""
    for j in range(i - 1, max(lo, i - 4) - 1, -1):
        t = tok(words[j]['w'])
        if t in ('right', 'righty') and j > 0 and tok(words[j - 1]['w']) == 'all': return j - 1
        if CUE.match(t): return j
        if t not in ('and', 'so', 'great', 'good'): return None
    return None

def locate(words, a, b, quote):
    """Time of the best match for a quote within words[a:b], or None if nothing matches well."""
    from difflib import SequenceMatcher
    q = [tok(w) for w in quote.split() if tok(w)]
    if not q: return None
    seg = [tok(w['w']) for w in words[a:b]]
    best, at = 0.0, None
    first = set(q[:3])
    for i in range(len(seg) - len(q) + 1):
        if seg[i] not in first and seg[i + 1 if i + 1 < len(seg) else i] not in first: continue
        r = SequenceMatcher(None, seg[i:i + len(q)], q).ratio()
        if r > best: best, at = r, i
    return words[a + at]['s'] if best >= 0.7 else None


def find(words, a, b, turns):
    """For each rogue after the first, find the host's cue + name handoff, searching forward from the last one.
    A pair is only labeled if the previous rogue's own turn was confirmed by a handoff into it: otherwise that
    turn may not exist (an invented rogue) and its pick can't be trusted as the label."""
    out, cursor, confirmed = [], a, True  # the first rogue is invited by the host's "X, go first"
    for prev, nxt in zip(turns, turns[1:]):
        names = aliases(nxt['rogue']); hit = None
        for i in range(cursor, b):
            if tok(words[i]['w']) in names and (j := cue_before(words, i, cursor)) is not None:
                hit = (j, i); break
        if hit is None:
            out.append({'prev': prev, 'next': nxt, 'cue': None, 'unverified': not confirmed}); confirmed = False; continue
        j, i = hit; cursor = i + 1
        out.append({'prev': prev, 'next': nxt, 'cue': j if confirmed else None, 'name': i, 'unverified': not confirmed})
        confirmed = True
    return out


if __name__ == '__main__':
    source, nums = sys.argv[1], sys.argv[2:]
    eps = json.loads((D / 'episodes.json').read_text())
    rows = []
    for n in nums:
        if source != 'reference' and not (D / 'extract' / source / f'{n}.json').exists(): continue  # dropped upstream
        tx = json.loads((D / 'tx' / f'{n}.json').read_text())['words']; seg = json.loads((D / 'seg' / f'{n}.json').read_text())
        turns = turns_for(source, n); fic = eps[n]['fiction']
        for t in turns: t['pick_time'] = locate(tx, seg['a'], seg['b'], t.get('quote', ''))
        if source == 'consensus':  # early episodes have no site key; consensus carries the agreed reveal
            fic = json.loads((D / 'extract' / source / f'{n}.json').read_text())['fiction']
        for k, h in enumerate(find(tx, seg['a'], seg['b'], turns), 1):
            row = {'episode': n, 'handoff': k, 'prev_rogue': h['prev']['rogue'], 'prev_pick': h['prev']['pick'], 'fiction': fic,
                   'prev_correct': int(h['prev']['pick'] == fic), 'next_rogue': h['next']['rogue'], 'position': k}
            if h['unverified']:
                row.update(kind='previous turn unverified')
            elif h['cue'] is None:
                row.update(kind='not found')
            else:
                ws = tx[h['cue']:h['name'] + 1]
                cue_words = [w for w in ws if tok(w['w']) in ('okay', 'ok', 'all', 'right', 'righty', 'alright', 'alrighty')]
                # "All right" is two words; the cue ends at "right" when present
                cue_end = cue_words[-1]['e'] if cue_words else ws[0]['e']
                row.update(kind='cue', cue_text=''.join(w['w'] for w in ws).strip(), cue_start=ws[0]['s'], cue_end=cue_end, name_end=ws[-1]['e'])
                # the handoff must sit between the previous rogue's stated pick and the next rogue's
                tp, tn = h['prev'].get('pick_time'), h['next'].get('pick_time')
                row['order_ok'] = '' if source == 'reference' else int(tp is not None and tn is not None and tp < ws[0]['s'] < tn)
            rows.append(row)
    out = D / f'handoffs_{source}{SFX}.csv'
    keys = ['episode', 'handoff', 'position', 'prev_rogue', 'prev_pick', 'fiction', 'prev_correct', 'next_rogue', 'kind', 'cue_text', 'cue_start', 'cue_end', 'name_end', 'order_ok']
    with out.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    for r in rows: print(r['episode'], r['handoff'], r['kind'], repr(r.get('cue_text')), r.get('cue_start'), f"{r['prev_rogue']}->{r['next_rogue']}", 'correct' if r['prev_correct'] else 'wrong')
