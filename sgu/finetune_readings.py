"""Fine-tune a pretrained speech model to pick the fiction from the host's three item readings (see readings.py).
The network scores each reading and is trained with a softmax over each episode's three, so it learns which of
his three readings is the odd one out. Folds are 5-fold grouped by episode; 15% of training episodes pick the
best epoch. Reports how often it picks the fiction (chance 33%), alone and combined with the text-only model,
and appends to data/finetune_readings_report.txt. Out-of-fold scores go to data/finetune_readings_<backbone>.npy.
usage: finetune_readings.py [--backbone wavlm|emotion] [--epochs 6] [--seed 0]"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import BACKBONES, Net
D = Path(__file__).resolve().parent.parent / 'data'
SR, CROP = 16000, 12.0  # seconds; longer readings are cropped (randomly in training, centred otherwise)
dev = 'cuda'

def audio(rows):
    """All readings as int16, cached in data/readings_audio.npz (one array plus offsets)."""
    cache = D / 'readings_audio.npz'
    if cache.exists():
        z = np.load(cache)
        offsets = z['offsets']
        if len(offsets) == len(rows) + 1:
            pcm = z['pcm']  # read once: every z[...] access re-reads the array from the file
            return [pcm[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    out = []
    for r in rows:
        a, b = float(r['start']), float(r['end'])
        out.append(np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-t', str(b - a), '-i', str(D / 'audio' / f"{r['episode']}.mp3"),
                                                '-f', 's16le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout, np.int16))
    np.savez(cache, pcm=np.concatenate(out), offsets=np.cumsum([0] + [len(x) for x in out]))
    return out

def batch(pcm, idx, train, rng):
    n = int(CROP * SR); x = np.zeros((len(idx), n), np.float32); lens = []
    for j, i in enumerate(idx):
        w = pcm[i].astype(np.float32) / 32768
        if len(w) > n:
            s = rng.integers(0, len(w) - n + 1) if train else (len(w) - n) // 2; w = w[s:s + n]
        if train: w = w * 10 ** (rng.uniform(-6, 6) / 20) + rng.normal(0, 10 ** (-rng.uniform(30, 45) / 20) * (w.std() + 1e-6), len(w))
        w = (w - w.mean()) / (w.std() + 1e-6)
        x[j, :len(w)] = w; lens.append(len(w))
    return torch.from_numpy(x).to(dev), torch.tensor(lens, device=dev)

def scores(net, pcm, eps, ep_rows):
    net.eval(); out = {}
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        for k in range(0, len(eps), 8):
            idx = [i for e in eps[k:k + 8] for i in ep_rows[e]]
            for i, s in zip(idx, net(*batch(pcm, idx, False, None)).float().cpu().numpy()): out[i] = s
    return out

def nll(sc, eps, ep_rows, fic):
    return np.mean([-(sc[fic[e]] - np.log(np.sum(np.exp([sc[i] for i in ep_rows[e]])))) for e in eps])

def train_fold(name, pcm, ep_rows, fic, tr, va, epochs, seed, per_batch):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(name).to(dev); net.bb.gradient_checkpointing_enable()
    bb = [p for p in net.bb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 2e-5}, {'params': [net.mix, *net.head.parameters()], 'lr': 1e-3}], weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / per_batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    best, best_state, best_ep = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), per_batch):
            eps = order[k:k + per_batch]; idx = [i for e in eps for i in ep_rows[e]]
            target = torch.tensor([ep_rows[e].index(fic[e]) for e in eps], device=dev)
            with torch.autocast('cuda', torch.bfloat16):
                logits = net(*batch(pcm, idx, True, rng)).float().view(len(eps), 3)
            loss = nn.functional.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        v = nll(scores(net, pcm, va, ep_rows), va, ep_rows, fic)
        if v < best: best, best_ep, best_state = v, ep, {k: t.detach().to('cpu', copy=True) for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best, best_ep

def top1(sc, eps, ep_rows, fic): return np.mean([max(ep_rows[e], key=lambda i: sc[i]) == fic[e] for e in eps])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--backbone', default='wavlm', choices=BACKBONES)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--seed', type=int, default=0); args = ap.parse_args()
    rows = list(csv.DictReader(open(D / 'readings.csv'))); pcm = audio(rows)
    ep_rows, fic = {}, {}
    for i, r in enumerate(rows):
        ep_rows.setdefault(r['episode'], []).append(i)
        if r['is_fiction'] == '1': fic[r['episode']] = i
    eps = np.array(sorted(ep_rows, key=int)); fic_item = np.array([int(rows[fic[e]]['item']) for e in eps])
    per_batch = 4 if args.backbone == 'emotion' else 8
    sc, t0 = {}, time.time()
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(eps, fic_item, eps)):
        cache = D / f'finetune_readings_{args.backbone}_e{args.epochs}_s{args.seed}_fold{f}.json'  # finished folds are kept, so a restart resumes
        if cache.exists():
            s = {int(k): v for k, v in json.loads(cache.read_text()).items()}; sc.update(s)
            print(f'fold {f}: cached, picks the fiction in {top1(s, eps[te], ep_rows, fic):.1%} of test episodes', flush=True); continue
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=args.seed).split(tr, groups=eps[tr]))
        net, v, ep = train_fold(BACKBONES[args.backbone], pcm, ep_rows, fic, eps[tr][a], eps[tr][b], args.epochs, args.seed + f, per_batch)
        s = scores(net, pcm, eps[te], ep_rows); sc.update(s)
        cache.write_text(json.dumps({str(k): float(v) for k, v in s.items()}))
        print(f'fold {f}: best epoch {ep + 1}, validation loss {v:.3f} (chance {np.log(3):.3f}), picks the fiction in '
              f'{top1(s, eps[te], ep_rows, fic):.1%} of test episodes ({time.time() - t0:.0f} s)', flush=True)
        del net; torch.cuda.empty_cache()
    np.save(D / f'finetune_readings_{args.backbone}_s{args.seed}.npy', np.array([sc[i] for i in range(len(rows))]))
    # combined with the text-only model: add its out-of-fold log-score to the network's (both are per-reading scores)
    from classify import centre, oof
    E = json.loads((D / 'episodes.json').read_text())
    y = np.array([int(r['is_fiction']) for r in rows]); groups = np.array([int(r['episode']) for r in rows])
    txt = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [E[r['episode']]['items'][int(r['item']) - 1]['text'] for r in rows]).toarray()
    ts = np.mean([oof(0, centre(txt, groups), y, groups, 50 + s) for s in range(3)], 0)
    both = {i: sc[i] + np.log(ts[i] / (1 - ts[i])) for i in range(len(rows))}
    line = (f'{args.backbone} {args.epochs} epochs s{args.seed}: picks the fiction {top1(sc, eps, ep_rows, fic):.1%} alone, '
            f'{top1(both, eps, ep_rows, fic):.1%} with the text model (text alone {top1(dict(enumerate(ts)), eps, ep_rows, fic):.1%}); chance 33.3%')
    print(line)
    with open(D / 'finetune_readings_report.txt', 'a') as fh: fh.write(line + '\n')
