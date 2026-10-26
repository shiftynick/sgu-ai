"""Does the host's "Okay" tell you whether the guess he just heard was right?
Trains logistic-regression classifiers on data/feat/ (see features.py) and scores them with cross-validation
grouped by episode, so no episode is in both training and test data.

The label is confounded with things that aren't the voice: later rogues are right more often (they hear the
others' reasoning), and "All right" turns up more often after a correct guess than "Okay". So `meta` is a
baseline from position, cue word, year and which rogue answered, with no audio at all, and each audio or text
feature set X is judged by whether meta+X beats meta. Its p-value comes from shuffling X between clips with the
same position, cue word and era: that keeps whatever X says about those and breaks only the rest.

Two analyses:
- all clips: AUC of X alone and of meta+X
- within episode: only episodes with a cue after a right guess and one after a wrong guess, features centred
  within each episode, scored as how often the model ranks the after-right cue above the after-wrong one.
  Recording, era and the host's mood that day are identical within a pair, so this is the cleanest test.
Writes data/classify_report.txt.
usage: classify.py [--perms 100]"""
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ADV, CONTEXT, PROSODY
warnings.filterwarnings('ignore')
D = Path(__file__).resolve().parent.parent / 'data'
WAVLM_LAYERS = slice(4, 10)  # mid layers carry the most paralinguistic information in WavLM; fixed in advance, not tuned
ROGUES = ['Bob', 'Jay', 'Evan', 'Rebecca', 'Cara']
WIDE = 32  # embeddings and text are reduced to this many PCA components inside each fold

def load():
    rows = list(csv.DictReader(open(D / 'dataset.csv')))
    feat, text = {}, {}
    for n in {r['episode'] for r in rows}:
        if not (D / 'feat' / f'{n}.npz').exists(): continue
        z = dict(np.load(D / 'feat' / f'{n}.npz')); t = json.loads((D / 'feat' / f'{n}.json').read_text())
        for i, h in enumerate(z['handoff']):
            feat[n, str(h)] = {k: v[i].astype(np.float32) for k, v in z.items() if k not in ('handoff', 'version')}
            text[n, str(h)] = t[str(h)]
    rows = [r for r in rows if (r['episode'], r['handoff']) in feat]
    f = lambda k: np.array([feat[r['episode'], r['handoff']][k] for r in rows])
    y = np.array([int(r['label_prev_correct']) for r in rows])
    groups = np.array([int(r['episode']) for r in rows])
    word = np.array([0 if r['cue_text'].lower().startswith('ok') else 1 for r in rows])
    pos = np.array([min(int(r['position']), 3) for r in rows])
    year = np.array([int(r['date'].split()[-1]) for r in rows])
    meta = np.column_stack([pos == 1, pos == 2, word, year] + [[r['prev_rogue'] == g for r in rows] for g in ROGUES]).astype(float)
    # host words during the turn and the cue words, kept apart so "okay" in his comment and in the cue differ
    docs = [' '.join(['h_' + w for w in a.lower().split()] + ['c_' + w for w in b.lower().split()])
            for a, b in (text[r['episode'], r['handoff']].split('||') for r in rows)]
    docs = [''.join(ch for ch in d if ch.isalnum() or ch in " _'") for d in docs]
    tf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), token_pattern=r"[a-z_']+")
    txt = tf.fit_transform(docs).toarray()  # vocabulary and idf use no labels, so fitting them on everything is fine
    sets = {'prosody': f('prosody'), 'context': f('context'), 'emotion (A/D/V)': np.hstack([f('adv'), f('adv_name')]),
            'wavlm cue': f('wavlm')[:, WAVLM_LAYERS].mean(1), 'wavlm cue+name': f('wavlm_name')[:, WAVLM_LAYERS].mean(1),
            'emo-emb cue': f('emo'), 'emo-emb cue+name': f('emo_name'), 'text': txt}
    sets['prosody+context'] = np.hstack([sets['prosody'], sets['context']])
    era = np.digitize(year, [2010, 2014, 2018, 2022])  # five eras of about four years
    return rows, y, groups, (era * 3 + pos - 1) * 2 + word, pos, meta, sets, (tf, txt)

def model(n_meta, X):
    """Logistic regression; wide blocks get PCA. The meta columns (first n_meta) are only scaled."""
    wide = X.shape[1] - n_meta > 64
    xpipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), *([PCA(WIDE, random_state=0)] if wide else []))
    parts = [('x', xpipe, list(range(n_meta, X.shape[1])))]
    if n_meta: parts.append(('meta', StandardScaler(), list(range(n_meta))))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=0.1, max_iter=2000))

def oof(n_meta, X, y, groups, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        p[te] = model(n_meta, X).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

def centre(X, groups):
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    out = X.copy()
    for g in np.unique(groups): i = groups == g; out[i] -= X[i].mean(0)
    return out

def pair_acc(p, y, groups):
    """Share of (after-right, after-wrong) cue pairs within an episode where the after-right one scores higher."""
    s = []
    for g in np.unique(groups):
        i = groups == g; a, b = p[i][y[i] == 1], p[i][y[i] == 0]
        s += [(x > z) + 0.5 * (x == z) for x in a for z in b]
    return np.mean(s)

def score(kind, n_meta, X, y, groups, seeds):
    if kind == 'all': return np.mean([roc_auc_score(y, oof(n_meta, X, y, groups, s)) for s in seeds])
    return np.mean([pair_acc(oof(n_meta, X, y, groups, s), y, groups) for s in seeds])

def shuffle_rows(X, strata, rng):
    X = X.copy()
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s); X[i] = X[rng.permutation(i)]
    return X

def test(kind, base, X, y, groups, strata, perms, seeds, prep=lambda A: A):
    """Observed score of meta+X, and the share of within-strata shuffles of X that score as well."""
    obs = score(kind, base.shape[1], prep(np.hstack([base, X])), y, groups, seeds)
    def one(k):
        Xs = shuffle_rows(X, strata, np.random.default_rng(k))
        return score(kind, base.shape[1], prep(np.hstack([base, Xs])), y, groups, seeds[:1])
    null = np.array(Parallel(n_jobs=-1)(delayed(one)(k) for k in range(perms)))
    return obs, null.mean(), (1 + (null >= obs).sum()) / (1 + perms)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--perms', type=int, default=100); args = ap.parse_args()
    rows, y, groups, strata, pos, meta, sets, (tf, txt) = load()
    seeds = range(5)
    out = [f'{len(y)} clips from {len(set(groups))} episodes: {y.sum()} after a correct guess, {len(y) - y.sum()} after a wrong one',
           'AUC and pair accuracy: 0.5 is chance. p tests whether adding X to the no-audio baseline helps (see the top of classify.py).', '']
    def emit(s): out.append(s); print(s, flush=True)

    emit('All clips: cross-validated AUC (5-fold grouped by episode, mean over 5 shuffles)')
    m = score('all', meta.shape[1], meta, y, groups, seeds)
    emit(f'  {"meta (no audio)":18s} {m:.3f}')
    emit(f'  {"X":18s} {"X alone":>8s} {"meta+X":>8s} {"shuffled":>9s}   p')
    for name, X in sets.items():
        alone = score('all', 0, X, y, groups, seeds)
        obs, null, p = test('all', meta, X, y, groups, strata, args.perms, seeds)
        emit(f'  {name:18s} {alone:8.3f} {obs:8.3f} {null:9.3f}   {p:.3f}')

    mixed = np.array([len(set(y[groups == g])) == 2 for g in groups])
    yy, gg, ss = y[mixed], groups[mixed], strata[mixed]
    pos_meta = np.column_stack([pos == 1, pos == 2, strata % 2]).astype(float)[mixed]  # position and cue word: what varies within an episode
    prep = lambda A: centre(A, gg)
    emit(''); emit(f'Within episode: {mixed.sum()} clips from {len(set(gg))} episodes with both a right and a wrong guess before a cue')
    emit(f'  {"position+cue word":18s} {score("pair", 3, prep(pos_meta), yy, gg, seeds):.3f}')
    emit(f'  {"X":18s} {"X alone":>8s} {"pos+X":>8s} {"shuffled":>9s}   p')
    for name, X in sets.items():
        alone = score('pair', 0, prep(X[mixed]), yy, gg, seeds)
        obs, null, p = test('pair', pos_meta, X[mixed], yy, gg, ss, args.perms, seeds, prep)
        emit(f'  {name:18s} {alone:8.3f} {obs:8.3f} {null:9.3f}   {p:.3f}')
    emit(f'  ({len(sets)} feature sets in each table, so expect about one p < 0.10 by chance in each)')

    k = len(out)
    out += ['', 'Single features, after a correct vs a wrong guess (pitch in semitones and loudness in dB re the host\'s reading voice)',
            f'  {"feature":16s} {"correct":>8s} {"wrong":>8s}   p (Mann-Whitney)']
    named = [(f, sets['prosody'][:, j]) for j, f in enumerate(PROSODY)] + [(f, sets['context'][:, j]) for j, f in enumerate(CONTEXT)] + \
            [(f, sets['emotion (A/D/V)'][:, j]) for j, f in enumerate(ADV)] + [(f + '_name', sets['emotion (A/D/V)'][:, 3 + j]) for j, f in enumerate(ADV)]
    for f, v in named:
        a, b = v[y == 1], v[y == 0]; a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        out.append(f'  {f:16s} {np.median(a):8.3f} {np.median(b):8.3f}   {mannwhitneyu(a, b).pvalue:.3f}')
    out.append(f'  ({len(named)} features tested, so expect one or two p < 0.05 by chance)')

    lr = LogisticRegression(C=1, max_iter=2000).fit(txt, y)
    vocab = np.array(tf.get_feature_names_out()); order = np.argsort(lr.coef_[0])
    out += ['', 'Words most associated with a correct guess before (h_ = host during the turn, c_ = the cue):',
            '  ' + ', '.join(vocab[order[::-1][:15]]), 'and with a wrong guess before:', '  ' + ', '.join(vocab[order[:15]])]
    (D / 'classify_report.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out[k:]))
