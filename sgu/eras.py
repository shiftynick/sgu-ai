"""Did the host's tell change over the years? Splits the clips into five eras of about four years and, per era:
- how well a model trained on every era does on that era's clips (out-of-fold, so a shared tell shows up everywhere)
- how well a model trained on that era alone does, with the same within-strata shuffle test as classify.py
- the emotion model's arousal/dominance/valence for "Okay, <name>", after a right vs a wrong guess
Writes data/eras_report.txt.
usage: eras.py [--perms 100]"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import D, load, oof, score, test
from features import ADV
SETS = ['emotion (A/D/V)', 'wavlm cue+name', 'emo-emb cue+name']
EDGES = [2010, 2014, 2018, 2022]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--perms', type=int, default=100); args = ap.parse_args()
    rows, y, groups, strata, pos, meta, sets, _ = load()
    year = np.array([int(r['date'].split()[-1]) for r in rows])
    era = np.digitize(year, EDGES)
    names = [f'{a}-{b - 1}' for a, b in zip([year.min()] + EDGES, EDGES + [year.max() + 1])]
    out = []
    def emit(s): out.append(s); print(s, flush=True)

    emit('Model trained on all eras, scored per era (AUC, out-of-fold, mean over 5 shuffles)')
    preds = {'meta': [oof(meta.shape[1], meta, y, groups, s) for s in range(5)]}
    for k in SETS: preds[k] = [oof(meta.shape[1], np.hstack([meta, sets[k]]), y, groups, s) for s in range(5)]
    emit(f'  {"era":10s} {"clips":>6s} {"right":>6s} {"meta":>6s} ' + ' '.join(f'{"+" + k:>18s}' for k in SETS))
    for e, nm in enumerate(names):
        i = era == e
        auc = {k: np.mean([roc_auc_score(y[i], p[i]) for p in v]) for k, v in preds.items()}
        emit(f'  {nm:10s} {i.sum():6d} {y[i].mean():6.0%} {auc["meta"]:6.3f} ' + ' '.join(f'{auc[k]:18.3f}' for k in SETS))

    emit(''); emit('Model trained on each era alone: meta, then meta+X with its shuffle p-value')
    for e, nm in enumerate(names):
        i = era == e
        m = score('all', meta.shape[1], meta[i], y[i], groups[i], range(5))
        cells = []
        for k in SETS:
            obs, null, p = test('all', meta[i], sets[k][i], y[i], groups[i], strata[i], args.perms, range(5))
            cells.append(f'{obs:.3f} (p {p:.2f})')
        emit(f'  {nm:10s} meta {m:.3f}   ' + '   '.join(f'+{k}: {c}' for k, c in zip(SETS, cells)))

    emit(''); emit('Emotion model on "Okay, <name>": mean after a right guess minus after a wrong one (p, Mann-Whitney)')
    adv = sets['emotion (A/D/V)'][:, 3:]
    for e, nm in enumerate(names):
        i = era == e
        emit(f'  {nm:10s} ' + '   '.join(f'{a}: {adv[i & (y == 1), j].mean() - adv[i & (y == 0), j].mean():+.4f} '
                                         f'(p {mannwhitneyu(adv[i & (y == 1), j], adv[i & (y == 0), j]).pvalue:.3f})' for j, a in enumerate(ADV)))
    emit(f'  (for scale: the standard deviations are ' + ', '.join(f'{a} {adv[:, j].std():.3f}' for j, a in enumerate(ADV)) + ')')
    (D / 'eras_report.txt').write_text('\n'.join(out) + '\n')
