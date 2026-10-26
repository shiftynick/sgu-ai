"""Fine-tune a pretrained speech model to hear whether the guess before the host's "Okay" was right.
Same folds as classify.py (5-fold, grouped by episode); within each training fold 15% of episodes are held out
to pick the best epoch. Writes out-of-fold probabilities to data/finetune_<backbone>_<clip>.npy, then reports:
- AUC of the network alone
- AUC of the no-audio baseline (position, cue word, year, rogue) with and without the network's output added,
  with a 95% interval for the gain from resampling episodes
Appends to data/finetune_report.txt.
usage: finetune.py [--backbone wavlm|emotion] [--clip cue_name|cue] [--epochs 15] [--seed 0]"""
import argparse, copy, csv, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
D = Path(__file__).resolve().parent.parent / 'data'
SR, MAXLEN = 16000, 2.2  # seconds; 99% of "Okay, <name>" clips are shorter
BACKBONES = {'wavlm': 'microsoft/wavlm-base-plus', 'emotion': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'}
dev = 'cuda'

class Net(nn.Module):
    """Backbone with a frozen convolutional front end, a learned mix of its layers, masked mean-pooling, and a linear head."""
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(name, layerdrop=0.0)  # layerdrop would change how many layers the mix sees
        self.bb.feature_extractor._freeze_parameters()
        n = self.bb.config.num_hidden_layers + 1
        self.mix = nn.Parameter(torch.zeros(n))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.bb.config.hidden_size, 1))
    def forward(self, x, lens):
        hs = torch.stack(self.bb(x, output_hidden_states=True).hidden_states)          # (layers, B, T, H)
        h = (torch.softmax(self.mix, 0)[:, None, None, None] * hs).sum(0)
        frames = (lens.float() / x.shape[1] * h.shape[1]).ceil().clamp(min=1)
        mask = (torch.arange(h.shape[1], device=h.device)[None] < frames[:, None]).float()
        return self.head((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).squeeze(1)

def batches(wavs, idx, bs, train, rng):
    idx = rng.permutation(idx) if train else idx
    for k in range(0, len(idx), bs):
        b = idx[k:k + bs]; n = int(MAXLEN * SR)
        x = np.zeros((len(b), n), np.float32); lens = []
        for j, i in enumerate(b):
            w = wavs[i][:n]
            if train:  # gain +-6 dB, a little noise, a small shift
                w = w * 10 ** (rng.uniform(-6, 6) / 20); w = w + rng.normal(0, 10 ** (-rng.uniform(30, 45) / 20) * (w.std() + 1e-6), len(w))
                off = rng.integers(0, max(1, min(int(0.15 * SR), n - len(w)) + 1)); w = w[:n - off]
            else: off = 0
            w = (w - w.mean()) / (w.std() + 1e-6)  # what the backbones' feature extractors do
            x[j, off:off + len(w)] = w; lens.append(off + len(w))
        yield b, torch.from_numpy(x).to(dev), torch.tensor(lens, device=dev)

def predict(net, wavs, idx):
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for b, x, lens in batches(wavs, idx, 64, False, None): out.append(torch.sigmoid(net(x, lens).float()).cpu().numpy())
    return np.concatenate(out)

def train_fold(name, wavs, y, tr, va, epochs, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(name).to(dev)
    bb = [p for p in net.bb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 2e-5}, {'params': [net.mix, *net.head.parameters()], 'lr': 1e-3}], weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / 32)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    pos_w = torch.tensor((y[tr] == 0).sum() / (y[tr] == 1).sum(), device=dev)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        for b, x, lens in batches(wavs, tr, 32, True, rng):
            with torch.autocast('cuda', torch.bfloat16):
                loss = nn.functional.binary_cross_entropy_with_logits(net(x, lens).float(), torch.tensor(y[b], device=dev, dtype=torch.float), pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        auc = roc_auc_score(y[va], predict(net, wavs, va))
        if auc > best: best, best_ep, best_state = auc, ep, {k: v.detach().to('cpu', copy=True) for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best, best_ep

def meta_matrix(rows):
    from classify import ROGUES
    word = np.array([0 if r['cue_text'].lower().startswith('ok') else 1 for r in rows])
    pos = np.array([min(int(r['position']), 3) for r in rows]); year = np.array([int(r['date'].split()[-1]) for r in rows])
    return np.column_stack([pos == 1, pos == 2, word, year] + [[r['prev_rogue'] == g for r in rows] for g in ROGUES]).astype(float)

def stacked(X, y, groups, seeds=range(5)):
    """Out-of-fold probabilities of a logistic regression on X, averaged over fold shuffles."""
    ps = []
    for s in seeds:
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=100 + s).split(X, y, groups):
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, 0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='wavlm', choices=BACKBONES); ap.add_argument('--clip', default='cue_name', choices=['cue_name', 'cue'])
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--seed', type=int, default=0); args = ap.parse_args()
    rows = list(csv.DictReader(open(D / 'dataset.csv')))
    y = np.array([int(r['label_prev_correct']) for r in rows]); groups = np.array([int(r['episode']) for r in rows])
    wavs = [sf.read(str(D / r['clip_with_name' if args.clip == 'cue_name' else 'clip']), dtype='float32')[0] for r in rows]
    p = np.zeros(len(y)); t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(wavs, y, groups)):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=args.seed).split(tr, y[tr], groups[tr]))
        net, va_auc, ep = train_fold(BACKBONES[args.backbone], wavs, y, tr[a], tr[b], args.epochs, args.seed + f)
        p[te] = predict(net, wavs, te)
        print(f'fold {f}: best epoch {ep + 1}, validation AUC {va_auc:.3f}, test AUC {roc_auc_score(y[te], p[te]):.3f} ({time.time() - t0:.0f} s)', flush=True)
        del net; torch.cuda.empty_cache()
    tag = f'{args.backbone}_{args.clip}_s{args.seed}'
    np.save(D / f'finetune_{tag}.npy', p)
    meta = meta_matrix(rows); logit = np.log(p / (1 - p + 1e-9) + 1e-9)[:, None]
    pm, pmn = stacked(meta, y, groups), stacked(np.hstack([meta, logit]), y, groups)
    # 95% interval for the gain, resampling whole episodes
    eps = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in eps}; rng = np.random.default_rng(0); gains = []
    for _ in range(2000):
        i = np.concatenate([idx[g] for g in rng.choice(eps, len(eps))])
        if len(set(y[i])) == 2: gains.append(roc_auc_score(y[i], pmn[i]) - roc_auc_score(y[i], pm[i]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    line = (f'{tag}: network alone AUC {roc_auc_score(y, p):.3f}; meta {roc_auc_score(y, pm):.3f}, meta+network {roc_auc_score(y, pmn):.3f}, '
            f'gain {roc_auc_score(y, pmn) - roc_auc_score(y, pm):+.3f} (95% interval {lo:+.3f} to {hi:+.3f})')
    print(line)
    with open(D / 'finetune_report.txt', 'a') as fh: fh.write(line + '\n')
