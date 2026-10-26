"""Does the host give away the fiction while reading the three items?
Finds each item reading in the transcript by matching the item text from theskepticsguide.org, then measures it.
Writes data/readings.csv (one row per reading: episode, item, is_fiction, timing and features) and
data/readings_emb.npz (WavLM and emotion-model embeddings, same row order).
Features are things that don't depend on what the item says:
- pace: words per second, pauses longer than 0.3 s, share of time in pauses
- hesitation: "uh"/"um" count, Whisper's mean and lowest word confidence, how far he strays from the written text
- voice: pitch level, spread, range and slope, the pitch fall at the end, loudness, harmonics-to-noise ratio,
  all relative to his median over the episode's three readings
- the emotion model's arousal, dominance and valence
usage: readings.py"""
import csv, json, re, sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import contours, emotion, load, wavlm_layers
D = Path(__file__).resolve().parent.parent / 'data'
SR = 16000
FEATS = ['dur', 'n_words', 'n_script', 'words_per_sec', 'n_pauses', 'pause_frac', 'fillers', 'conf_mean', 'conf_min', 'script_match',
         'f0_mean', 'f0_sd', 'f0_range', 'f0_slope', 'f0_end_drop', 'int_mean', 'int_sd', 'hnr', 'arousal', 'dominance', 'valence']

def tok(w): return re.sub(r"[^a-z0-9']", '', w.lower())

def find(words, lo, hi, text):
    """(first word index, last word index, match ratio) of the earliest good match of text in words[lo:hi]."""
    q = [tok(w) for w in text.split() if tok(w)]
    seg = [tok(w['w']) for w in words]
    best = (0, None)
    for i in range(lo, max(lo, hi - len(q) // 2)):
        r = SequenceMatcher(None, q, seg[i:i + len(q)], autojunk=False).ratio()
        if r > best[0]: best = (r, i)
        if r > 0.75: break  # the first strong match is him reading it; later ones are rogues quoting it
    r, i = best
    if i is None or r < 0.6: return None
    win = seg[i:i + len(q) + 10]
    blocks = [b for b in SequenceMatcher(None, q, win, autojunk=False).get_matching_blocks() if b.size]
    return i + blocks[0].b, i + blocks[-1].b + blocks[-1].size - 1, r

def measure(x, ws, script_len, ratio):
    dur = ws[-1]['e'] - ws[0]['s']
    gaps = np.array([b['s'] - a['e'] for a, b in zip(ws, ws[1:])])
    f0, ft, db, _, hnr = contours(x)
    v = f0 > 0
    tail = ft[v] > ft[-1] - 1.0 if v.sum() else np.array([], bool)
    fit = lambda t, y: np.polyfit(t, y, 1)[0] if len(y) > 2 else np.nan
    st = 12 * np.log2(f0[v]) if v.sum() else np.array([np.nan])  # absolute for now, centred on the episode later
    return {'dur': dur, 'n_words': len(ws), 'n_script': script_len, 'words_per_sec': len(ws) / dur,
            'n_pauses': int((gaps > 0.3).sum()), 'pause_frac': gaps[gaps > 0.3].sum() / dur,
            'fillers': sum(tok(w['w']) in ('uh', 'um', 'er', 'ah') for w in ws),
            'conf_mean': float(np.mean([w['p'] for w in ws])), 'conf_min': float(np.min([w['p'] for w in ws])), 'script_match': ratio,
            'f0_mean': np.mean(st), 'f0_sd': np.std(st), 'f0_range': np.percentile(st, 95) - np.percentile(st, 5) if v.sum() else np.nan,
            'f0_slope': fit(ft[v], st), 'f0_end_drop': np.mean(st) - np.mean(st[tail]) if tail.sum() else np.nan,
            'int_mean': np.mean(db[db > 0]), 'int_sd': np.std(db[db > 0]), 'hnr': np.mean(hnr[hnr > -200])}

if __name__ == '__main__':
    E = json.loads((D / 'episodes.json').read_text())
    rows, W, M = [], [], []
    for n, e in sorted(E.items(), key=lambda kv: int(kv[0])):
        items = e.get('items') or []
        if not (e.get('ok') and len(items) == 3 and all(it['text'] for it in items) and str(e.get('fiction')) in '123'): continue
        if not (D / 'seg' / f'{n}.json').exists(): continue
        seg = json.loads((D / 'seg' / f'{n}.json').read_text()); words = json.loads((D / 'tx' / f'{n}.json').read_text())['words']
        lo, spans = seg['a'], []
        for it in sorted(items, key=lambda it: it['num']):
            m = find(words, lo, seg['b'], it['text'])
            if not m: break
            spans.append((it['num'], *m)); lo = m[1] + 1
        if len(spans) < 3: print(n, 'items not all found', flush=True); continue
        ep = []
        for num, i, j, r in spans:
            ws = words[i:j + 1]; a, b = ws[0]['s'] - 0.1, ws[-1]['e'] + 0.2
            if not 3 <= b - a <= 45: break
            x = load(n, a, b)
            f = measure(x, ws, len(items[num - 1]['text'].split()), r)
            emb, adv = emotion(x); f.update(zip(['arousal', 'dominance', 'valence'], adv.tolist()))
            ep.append(({'episode': n, 'date': e['date'], 'item': num, 'is_fiction': int(str(num) == str(e['fiction'])),
                        'start': round(a, 2), 'end': round(b, 2), **{k: round(float(f[k]), 4) for k in FEATS}},
                       wavlm_layers(x)[4:10].mean(0), emb))
        if len(ep) < 3: print(n, 'implausible reading length', flush=True); continue
        for k in ('f0_mean',):  # pitch level relative to his median over this episode's readings
            med = np.median([r[k] for r, _, _ in ep])
            for r, _, _ in ep: r[k] = round(r[k] - med, 4)
        for r, w, m in ep: rows.append(r); W.append(w); M.append(m)
        if len(rows) % 300 == 0: print(len(rows) // 3, 'episodes', flush=True)
    with open(D / 'readings.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    np.savez(D / 'readings_emb.npz', wavlm=np.array(W, np.float16), emo=np.array(M, np.float16))
    print(f'{len(rows) // 3} episodes, {len(rows)} readings')
