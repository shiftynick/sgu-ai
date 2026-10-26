"""Compute features for every clip in data/dataset.csv, cached per episode in data/feat/<n>.npz (+ <n>.json for text).
Each cue is measured against the host's own voice in that episode (30 s of him reading the items),
so recording quality and era drop out.
- prosody: duration, pitch and loudness contour of the cue word, the pause before it, the name after it
- context: how much the host spoke during the rogue's turn, found by matching 1.5 s windows to his voice
- wavlm, wavlm_name: WavLM-base-plus hidden states of the cue and of "cue + name", mean-pooled per layer,
  minus the same for the host reference
- emo, emo_name: an emotion-tuned wav2vec2 (audeering, MSP-Podcast) on the same two clips: its pooled
  embedding minus the host reference, plus its arousal/dominance/valence estimates
- text (in the .json): the words Whisper heard in the host-voiced windows of the turn, then the cue and name
usage: features.py [--redo]"""
import csv, glob, json, re, subprocess, sys
from pathlib import Path
import numpy as np, parselmouth, torch, torch.nn as nn
D = Path(__file__).resolve().parent.parent / 'data'
SR = 16000
VERSION = 2  # bump to recompute every episode after changing what is extracted
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
PROSODY = ['dur', 'pause_before', 'name_gap', 'voiced_frac', 'f0_mean', 'f0_sd', 'f0_min', 'f0_max', 'f0_range', 'f0_start', 'f0_end',
           'f0_slope', 'f0_peak_pos', 'int_mean', 'int_max', 'int_slope', 'int_peak_pos', 'hnr']
CONTEXT = ['host_secs', 'host_frac', 'host_words', 'host_secs_last5']
ADV = ['arousal', 'dominance', 'valence']

def norm(w): return re.sub(r"[^a-z']", '', w.lower())

def load(n, a, b):
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(max(a, 0)), '-t', str(b - max(a, 0)), '-i', str(D / 'audio' / f'{n}.mp3'),
                          '-f', 's16le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float64) / 32768

def host_ref_start(words, seg):
    """Same reference window as cut.py: where the host starts reading the first item."""
    i = next((i for i in range(seg['a'], seg['b']) if norm(words[i]['w']) == 'item'), None)
    return words[i]['s'] if i is not None else seg['start'] + 5

def contours(x):
    snd = parselmouth.Sound(x, SR)
    p = snd.to_pitch_ac(time_step=0.005, pitch_floor=60, pitch_ceiling=400)
    f0 = p.selected_array['frequency']
    it = snd.to_intensity(minimum_pitch=60, time_step=0.005)
    hnr = snd.to_harmonicity_cc(time_step=0.005, minimum_pitch=60).values[0]
    return f0, p.xs(), it.values[0], it.xs(), hnr

def prosody(x, ref_f0, ref_db, pause, gap):
    f0, ft, db, dt, hnr = contours(x)
    v = f0 > 0
    st = 12 * np.log2(f0[v] / ref_f0) if v.sum() else np.array([np.nan])  # semitones re the host's median pitch
    tv = ft[v]; dur = len(x) / SR
    k = max(1, len(st) // 5)
    fit = lambda t, y: np.polyfit(t, y, 1)[0] if len(y) > 2 else np.nan
    db = db - ref_db
    return [dur, pause, gap, v.mean(), np.mean(st), np.std(st), np.min(st), np.max(st), np.ptp(st), np.mean(st[:k]), np.mean(st[-k:]),
            fit(tv, st), tv[np.argmax(st)] / dur if v.sum() else np.nan, np.mean(db), np.max(db), fit(dt, db), dt[np.argmax(db)] / dur,
            np.mean(hnr[hnr > -200]) if (hnr > -200).any() else np.nan]

class RegressionHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense, self.dropout, self.out_proj = nn.Linear(c.hidden_size, c.hidden_size), nn.Dropout(c.final_dropout), nn.Linear(c.hidden_size, c.num_labels)
    def forward(self, x): return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

_models = {}
def models():
    """Loaded on first use, so classify.py can import this file cheaply."""
    if not _models:
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor, WavLMModel
        class EmotionModel(Wav2Vec2PreTrainedModel):  # the model card's head: arousal, dominance, valence
            def __init__(self, c):
                super().__init__(c); self.wav2vec2 = Wav2Vec2Model(c); self.classifier = RegressionHead(c); self.post_init()
            def forward(self, x):
                h = self.wav2vec2(x).last_hidden_state.mean(1); return h, self.classifier(h)
        name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        emo, info = EmotionModel.from_pretrained(name, output_loading_info=True)
        assert not info['missing_keys'], info['missing_keys']
        _models.update(fe=AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus'),
                       wavlm=WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(dev).eval(),
                       emo_proc=Wav2Vec2Processor.from_pretrained(name), emo=emo.to(dev).eval(),
                       spk=EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': dev}))
    return _models

def wavlm_layers(x):
    m = models()
    inp = m['fe'](x.astype(np.float32), sampling_rate=SR, return_tensors='pt').input_values.to(dev)
    with torch.inference_mode():
        hs = m['wavlm'](inp, output_hidden_states=True).hidden_states
    return torch.stack([h[0].mean(0) for h in hs]).cpu().numpy()  # (13, 768)

def emotion(x):
    m = models()
    inp = m['emo_proc'](x.astype(np.float32), sampling_rate=SR, return_tensors='pt').input_values.to(dev)
    with torch.inference_mode():
        h, adv = m['emo'](inp)
    return h[0].cpu().numpy(), adv[0].cpu().numpy()

def ecapa(xs):
    with torch.inference_mode():
        e = models()['spk'].encode_batch(torch.from_numpy(np.stack(xs)).float().to(dev)).squeeze(1)
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy()

def host_windows(n, a, b, host_ref):
    """Host-voiced 1.5 s windows (0.5 s hop) in [a, b]: closer to the host's reference voice than to the turn's
    average voice, which is mostly the rogue's. Returns window start times."""
    if b - a < 1.5: return []
    x = load(n, a, b)
    starts = np.arange(0, len(x) / SR - 1.5 + 1e-9, 0.5)
    e = ecapa([x[int(s * SR):int(s * SR) + int(1.5 * SR)] for s in starts])
    turn = e.mean(0); turn /= np.linalg.norm(turn)
    is_host = (e @ host_ref > e @ turn) & (e @ host_ref > 0.4)
    return [a + s for s, h in zip(starts, is_host) if h]

def merged(ws):
    """Total seconds covered by 1.5 s windows starting at ws."""
    total, end = 0.0, -1e9
    for s in sorted(ws):
        total += max(0.0, s + 1.5 - max(s, end)); end = max(end, s + 1.5)
    return total

if __name__ == '__main__':
    redo = '--redo' in sys.argv
    rows = list(csv.DictReader(open(D / 'dataset.csv')))
    # where each turn starts: the end of the previous handoff's name ("Okay, Bob" -> Bob's turn)
    name_end = {(r['episode'], r['handoff']): float(r['name_end'])
                for p in sorted(glob.glob(str(D / 'handoffs_consensus*.csv'))) for r in csv.DictReader(open(p)) if r['kind'] == 'cue'}
    (D / 'feat').mkdir(exist_ok=True)
    by_ep = {}
    for r in rows: by_ep.setdefault(r['episode'], []).append(r)
    done = 0
    for n, rs in by_ep.items():
        out = D / 'feat' / f'{n}.npz'
        hs = sorted(r['handoff'] for r in rs)
        if not redo and out.exists():
            z = np.load(out)
            if 'version' in z and int(z['version']) == VERSION and sorted(z['handoff'].tolist()) == hs: continue
        words = json.loads((D / 'tx' / f'{n}.json').read_text())['words']
        seg = json.loads((D / 'seg' / f'{n}.json').read_text())
        t0 = host_ref_start(words, seg)
        ref = load(n, t0, t0 + 30)
        f0, _, db, _, _ = contours(ref)
        ref_f0, ref_db = np.median(f0[f0 > 0]), np.median(db[db > 0])
        ref_w, (ref_e, _), ref_spk = wavlm_layers(ref), emotion(ref), ecapa([ref])[0]
        clips = {r['handoff']: r for r in rs}
        F = {k: [] for k in ('prosody', 'context', 'wavlm', 'wavlm_name', 'emo', 'emo_name', 'adv', 'adv_name')}
        text = {}
        for h in hs:
            r = clips[h]; cs, ce = float(r['cue_start']), float(r['cue_end'])
            # the pause before the cue: from the end of the last word Whisper heard before it
            prev = [w['e'] for w in words if w['s'] < cs - 0.1 and w['e'] <= cs + 0.05]
            pause = cs - prev[-1] if prev else np.nan
            x, xn = parselmouth.Sound(str(D / r['clip'])).values[0], parselmouth.Sound(str(D / r['clip_with_name'])).values[0]
            gap = (len(xn) - len(x)) / SR  # name and the gap before it, a proxy for how quickly he moves on
            F['prosody'].append(prosody(x, ref_f0, ref_db, pause, gap))
            # the rogue's turn: from the previous handoff (or at most 30 s back) to just before the cue
            a = max(name_end.get((n, str(int(h) - 1)), cs - 30), cs - 30) + 0.3
            hw = host_windows(n, a, cs - 0.3, ref_spk)
            hw_words = [w for w in words if a <= (w['s'] + w['e']) / 2 < cs - 0.3 and any(s <= (w['s'] + w['e']) / 2 < s + 1.5 for s in hw)]
            F['context'].append([merged(hw), merged(hw) / max(cs - 0.3 - a, 1e-6), len(hw_words), merged([s for s in hw if s >= cs - 5.3])])
            text[h] = ' '.join(w['w'].strip() for w in hw_words) + ' || ' + r['cue_text']  # cue_text: the clip re-transcribed on its own
            F['wavlm'].append(wavlm_layers(x) - ref_w); F['wavlm_name'].append(wavlm_layers(xn) - ref_w)
            (e, adv), (en, advn) = emotion(x), emotion(xn)
            F['emo'].append(e - ref_e); F['emo_name'].append(en - ref_e); F['adv'].append(adv); F['adv_name'].append(advn)
        np.savez(out, version=VERSION, handoff=np.array(hs), **{k: np.array(v, dtype=np.float16 if k.startswith(('wavlm', 'emo')) else np.float32)
                                                             for k, v in F.items()})
        (D / 'feat' / f'{n}.json').write_text(json.dumps(text))
        done += 1
        if done % 50 == 0: print(done, 'episodes', flush=True)
    print(f'features computed for {done} new episodes, {len(by_ep)} in the dataset')
