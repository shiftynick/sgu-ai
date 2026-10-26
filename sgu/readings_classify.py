"""Can you pick the fiction from how the host reads the three items? (features from readings.py)
Each episode has three readings and exactly one fiction, so a model scores the three and picks the highest;
chance is 1 in 3. Features are centred within each episode, so only differences between his three readings count.
The baseline is item position plus the written item's length (he may put the fiction in some slots more often,
and longer items take longer to read). Each feature set X is judged by whether baseline+X picks the fiction
more often than baseline alone; its p-value comes from shuffling X between readings of the same item number.
`text` uses the item wording, not his voice: it shows how guessable the fiction is from content alone, which a
voice model could partly pick up through the words.
Writes data/readings_report.txt.
usage: readings_classify.py [--perms 200]"""
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import centre, oof, shuffle_rows
from readings import FEATS
warnings.filterwarnings('ignore')
D = Path(__file__).resolve().parent.parent / 'data'
PACE = ['words_per_sec', 'n_pauses', 'pause_frac', 'fillers', 'conf_mean', 'conf_min', 'script_match']
VOICE = ['f0_mean', 'f0_sd', 'f0_range', 'f0_slope', 'f0_end_drop', 'int_mean', 'int_sd', 'hnr']
ADV = ['arousal', 'dominance', 'valence']

def top1(p, y, groups):
    """Share of episodes where the highest-scored reading is the fiction."""
    return np.mean([y[groups == g][np.argmax(p[groups == g])] for g in np.unique(groups)])

def score(base, X, y, groups, seeds):
    A = centre(np.hstack([base, X]), groups)
    return np.mean([top1(oof(base.shape[1], A, y, groups, s), y, groups) for s in seeds])

def test(base, X, y, groups, strata, perms, seeds):
    obs = score(base, X, y, groups, seeds)
    null = np.array(Parallel(n_jobs=-1)(delayed(lambda k: score(base, shuffle_rows(X, strata, np.random.default_rng(k)), y, groups, seeds[:1]))(k)
                                        for k in range(perms)))
    return obs, null.mean(), (1 + (null >= obs).sum()) / (1 + perms)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--perms', type=int, default=200); args = ap.parse_args()
    rows = list(csv.DictReader(open(D / 'readings.csv'))); emb = np.load(D / 'readings_emb.npz')
    E = json.loads((D / 'episodes.json').read_text())
    y = np.array([int(r['is_fiction']) for r in rows]); groups = np.array([int(r['episode']) for r in rows])
    item = np.array([int(r['item']) for r in rows])
    col = lambda ks: np.array([[float(r[k]) for k in ks] for r in rows])
    base = np.column_stack([item == 1, item == 2, col(['n_script'])]).astype(float)
    txt = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [E[r['episode']]['items'][int(r['item']) - 1]['text'] for r in rows]).toarray()
    sets = {'pace+hesitation': col(PACE), 'voice': col(VOICE), 'emotion (A/D/V)': col(ADV), 'all of the above': col(PACE + VOICE + ADV),
            'wavlm': emb['wavlm'].astype(np.float32), 'emotion embedding': emb['emo'].astype(np.float32), 'text (content, not voice)': txt}
    seeds = range(5)
    out = [f'{len(set(groups))} episodes, {len(y)} readings. Fiction by position: ' +
           ', '.join(f'item {k} {np.mean(item[y == 1] == k):.0%}' for k in (1, 2, 3)), '']
    def emit(s): out.append(s); print(s, flush=True)
    emit('How often the model picks the fiction (chance 33%; 5-fold grouped by episode, mean over 5 shuffles)')
    emit(f'  {"baseline (position, length)":28s} {score(base[:, :0], base, y, groups, seeds):.1%}')
    emit(f'  {"X":28s} {"X alone":>8s} {"base+X":>8s} {"shuffled":>9s}   p')
    for name, X in sets.items():
        alone = score(base[:, :0], X, y, groups, seeds)
        obs, null, p = test(base, X, y, groups, item, args.perms, seeds)
        emit(f'  {name:28s} {alone:8.1%} {obs:8.1%} {null:9.1%}   {p:.3f}')
    # the same with an out-of-fold text score in the baseline: does the voice add anything beyond the wording?
    ts = np.mean([oof(0, centre(txt, groups), y, groups, 50 + s) for s in range(3)], 0)
    base_t = np.column_stack([base, np.log(ts / (1 - ts))])
    emit(''); emit('Beyond the wording: baseline plus the text model\'s score')
    emit(f'  {"baseline + text score":28s} {score(base_t[:, :0], base_t, y, groups, seeds):.1%}')
    emit(f'  {"X":28s} {"":>8s} {"base+X":>8s} {"shuffled":>9s}   p')
    for name, X in sets.items():
        if name.startswith('text'): continue
        obs, null, p = test(base_t, X, y, groups, item, args.perms, seeds)
        emit(f'  {name:28s} {"":8s} {obs:8.1%} {null:9.1%}   {p:.3f}')

    year = np.array([int(r['date'].split()[-1]) for r in rows]); era = np.digitize(year, [2010, 2014, 2018, 2022])
    emit(''); emit('By era: how often baseline+text+X picks the fiction (model trained on all eras, scored per era)')
    pe = {k: np.mean([oof(base_t.shape[1], centre(np.hstack([base_t, sets[k]]), groups), y, groups, s) for s in seeds], 0)
          for k in ('all of the above', 'emotion embedding')}
    for e, nm in enumerate(['2005-2009', '2010-2013', '2014-2017', '2018-2021', '2022-2026']):
        i = era == e
        emit(f'  {nm:10s} {len(set(groups[i])):4d} episodes   ' + '   '.join(f'{k}: {top1(v[i], y[i], groups[i]):.1%}' for k, v in pe.items()))

    out += ['', 'Single features: fiction minus the mean of the two real items in the same episode (median, Wilcoxon p)']
    F = col(FEATS)
    for j, f in enumerate(FEATS):
        d = [F[(groups == g) & (y == 1), j][0] - F[(groups == g) & (y == 0), j].mean() for g in np.unique(groups)]
        d = np.array(d); d = d[~np.isnan(d)]
        out.append(f'  {f:14s} {np.median(d):+9.4f}   p {wilcoxon(d).pvalue:.3f}')
    out.append(f'  ({len(FEATS)} features tested, so expect one p < 0.05 by chance)')
    (D / 'readings_report.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out[out.index('Single features: fiction minus the mean of the two real items in the same episode (median, Wilcoxon p)'):]))
