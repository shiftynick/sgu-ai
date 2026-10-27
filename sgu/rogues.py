"""Two follow-ups to readings_classify.py and finetune_readings.py.
1. Combining models: average the per-reading scores of the best voice models (each as out-of-fold log-odds,
   centred within the episode and scaled to unit spread) and see how often the combination picks the fiction.
2. Do the rogues pick up the tell? Their picks and the voice model both lean towards the fiction, so plain
   agreement proves nothing. The clean test uses only the guesses where a rogue picked a real item: of the two real
   items, do they pick the one the voice model thought more fiction-like more often than half the time?
   The same test with the text model shows what a cue the rogues are known to use (the wording) looks like.
   First pickers are reported separately because later rogues often follow earlier ones.
Out-of-fold scores go to data/readings_oof.npz; results to data/rogues_report.txt.
usage: rogues.py"""
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import centre, oof
from readings_classify import ADV, PACE, VOICE, top1
warnings.filterwarnings('ignore')
D = Path(__file__).resolve().parent.parent / 'data'
logit = lambda p: np.log(p / (1 - p))

def oof_scores(rows, groups, y):
    cache = D / 'readings_oof.npz'
    if cache.exists(): return dict(np.load(cache))
    E = json.loads((D / 'episodes.json').read_text()); emb = np.load(D / 'readings_emb.npz')
    item = np.array([int(r['item']) for r in rows]); col = lambda ks: np.array([[float(r[k]) for k in ks] for r in rows])
    base = np.column_stack([item == 1, item == 2, col(['n_script'])]).astype(float)
    txt = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [E[r['episode']]['items'][int(r['item']) - 1]['text'] for r in rows]).toarray()
    run = lambda X, nb: np.mean([oof(nb, centre(np.hstack([base[:, :nb], X]), groups), y, groups, s) for s in range(5)], 0)
    out = {'baseline': logit(run(base[:, :0], 3)), 'pace': logit(run(col(PACE), 3)), 'pitch': logit(run(col(VOICE), 3)),
           'adv': logit(run(col(ADV), 3)), 'emotion': logit(run(emb['emo'].astype(np.float32), 3)), 'wavlm': logit(run(emb['wavlm'].astype(np.float32), 3)),
           'features': logit(run(col(PACE + VOICE + ADV), 3)), 'text': logit(run(txt, 0)),
           'finetuned': np.load(D / 'finetune_readings_wavlm_s0.npy')}
    # the same with the text model's out-of-fold score added to the baseline: what the voice adds beyond the wording
    base = np.column_stack([base, out['text']])
    out.update({'text_base': logit(run(base[:, :0], 4)), 'emotion_text': logit(run(emb['emo'].astype(np.float32), 4)),
                'features_text': logit(run(col(PACE + VOICE + ADV), 4))})
    np.savez(cache, **out)
    return out

def norm(s, groups):
    c = centre(s[:, None], groups)[:, 0]
    return c / c.std()

if __name__ == '__main__':
    rows = list(csv.DictReader(open(D / 'readings.csv')))
    y = np.array([int(r['is_fiction']) for r in rows]); groups = np.array([int(r['episode']) for r in rows])
    S = {k: norm(v, groups) for k, v in oof_scores(rows, groups, y).items()}
    S['baseline'] = S['baseline'] + 1e-9 * np.random.default_rng(0).normal(size=len(y))  # break exact ties
    out = []
    def emit(s): out.append(s); print(s, flush=True)
    emit(f'1. Combining models: how often the top-scored reading is the fiction ({len(set(groups))} episodes, chance 33.3%)')
    combos = [['baseline'], ['pace'], ['pitch'], ['adv'], ['emotion'], ['wavlm'], ['finetuned'], ['features'], ['text'], ['emotion', 'finetuned'], ['emotion', 'wavlm', 'finetuned'],
              ['emotion', 'wavlm', 'finetuned', 'features'], ['emotion', 'finetuned', 'text'], ['emotion', 'wavlm', 'finetuned', 'features', 'text'],
              ['text_base'], ['emotion_text'], ['features_text']]
    for c in combos:
        emit(f'  {" + ".join(c):45s} {top1(sum(S[k] for k in c), y, groups):.1%}')
    emit('  correlations between the models\' scores:')
    ks = ['emotion', 'wavlm', 'finetuned', 'features', 'text']; C = np.corrcoef([S[k] for k in ks])
    for i, k in enumerate(ks): emit(f'    {k:10s} ' + ' '.join(f'{C[i, j]:+.2f}' for j in range(len(ks))))
    voice = sum(S[k] for k in ['emotion', 'wavlm', 'finetuned'])

    # rogue picks, from the consensus extraction
    idx = {(r['episode'], int(r['item'])): i for i, r in enumerate(rows)}
    turns = []
    for f in (D / 'extract' / 'consensus').glob('*.json'):
        n = f.stem; x = json.loads(f.read_text())
        if (n, 1) not in idx: continue
        for t in x.get('turns', []):
            if t.get('pick') in (1, 2, 3): turns.append((n, int(t.get('pos') or 0), t['pick']))
    emit(''); emit(f'2. Do the rogues pick up the tell? {len(turns)} guesses in {len({t[0] for t in turns})} episodes with readings')
    right = np.mean([y[idx[(n, p)]] for n, _, p in turns])
    emit(f'  rogues pick the fiction {right:.1%} of the time; the combined voice model {top1(voice, y, groups):.1%}')
    for name, s in [('voice model (emotion + WavLM + fine-tuned)', voice), ('text model (wording only)', S['text'])]:
        emit(f'  {name}:')
        for label, keep in [('all guesses', lambda pos: True), ('first picker only', lambda pos: pos == 1)]:
            hit = tot = 0
            for n, pos, p in turns:
                if not keep(pos) or y[idx[(n, p)]]: continue
                real = [idx[(n, k)] for k in (1, 2, 3) if not y[idx[(n, k)]]]
                if len(real) != 2: continue
                hit += s[idx[(n, p)]] == max(s[real]); tot += 1
            bt = binomtest(hit, tot, 0.5)
            emit(f'    {label:18s} wrong guesses landed on the real item the model found more fiction-like: {hit}/{tot} = {hit / tot:.1%} '
                 f'(chance 50%, 95% CI {bt.proportion_ci().low:.1%}-{bt.proportion_ci().high:.1%}, p {bt.pvalue:.3f})')
    (D / 'rogues_report.txt').write_text('\n'.join(out) + '\n')

    # 3. the rogues' own record, on every Steve-hosted segment with clear picks (not only those with readings)
    E = json.loads((D / 'episodes.json').read_text()); recs = []
    for f in (D / 'extract' / 'consensus').glob('*.json'):
        x = json.loads(f.read_text()); e = E.get(f.stem)
        if not (e and e.get('ok') and str(e.get('fiction')) in '123' and x.get('turns') and 'Steve' in (x.get('host') or '')): continue
        year = int(e['date'].split()[-1])
        for t in x['turns']:
            if t.get('pick') in (1, 2, 3) and t.get('pos'):
                recs.append((f.stem, year, t['rogue'].split()[0], int(t['pos']), int(t['pick'] == int(e['fiction']))))
    pos_rate = {p: np.mean([r[4] for r in recs if r[3] == p]) for p in {r[3] for r in recs}}
    eras = [(2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021), (2022, 2026)]
    stats = {'episodes': len({r[0] for r in recs}), 'guesses': len(recs),
             'by_position': {p: [round(100 * pos_rate[p], 1), sum(r[3] == p for r in recs)] for p in sorted(pos_rate)}, 'rogues': {}}
    ep_res = {}
    for r in recs: ep_res.setdefault(r[0], []).append(r[4])
    stats['sweeps'] = {'all right': sum(all(v) for v in ep_res.values()), 'all wrong': sum(not any(v) for v in ep_res.values())}
    emit(''); emit(f'3. The rogues\' record: {stats["guesses"]} guesses in {stats["episodes"]} Steve-hosted episodes; '
                   f'everyone right in {stats["sweeps"]["all right"]}, everyone wrong in {stats["sweeps"]["all wrong"]}')
    emit('  right by guessing position: ' + ', '.join(f'{p}: {v[0]:.0f}% of {v[1]}' for p, v in stats['by_position'].items()))
    emit(f'  {"rogue":10s} {"guesses":>7s} {"right":>6s} {"vs position":>11s} {"first":>6s}  by era')
    for name in ['Bob', 'Jay', 'Evan', 'Rebecca', 'Cara', 'Perry']:
        m = [r for r in recs if r[2] == name]; first = [r[4] for r in m if r[3] == 1]
        st = {'guesses': len(m), 'right': round(100 * np.mean([r[4] for r in m]), 1),
              'vs_position': round(100 * np.mean([r[4] - pos_rate[r[3]] for r in m]), 1),
              'first': [round(100 * np.mean(first), 1), len(first)],
              'eras': [[round(100 * np.mean(v), 1), len(v)] if (v := [r[4] for r in m if a <= r[1] <= b]) else None for a, b in eras]}
        stats['rogues'][name] = st
        emit(f'  {name:10s} {st["guesses"]:7d} {st["right"]:5.1f}% {st["vs_position"]:+10.1f}% {st["first"][0]:5.1f}%  ' +
             ' '.join(f'{e[0]:.0f}%/{e[1]}' if e else '-' for e in st['eras']))
    (D / 'rogues_stats.json').write_text(json.dumps(stats, indent=1))
    (D / 'rogues_report.txt').write_text('\n'.join(out) + '\n')
