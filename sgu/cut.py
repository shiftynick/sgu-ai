"""Re-align each handoff cue with a forced aligner, check it is the host's voice, and cut clips.
Reads data/handoffs_<source>.csv, writes data/clips/<n>_h<k>_{cue,cue_name}.wav and data/clips_<source>.csv.
usage: cut.py <source>"""
import csv, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
from speechbrain.inference.speaker import EncoderClassifier
D = Path(__file__).resolve().parent.parent / 'data'
SR = 16000
# batch runs set SGU_TAG so each batch writes its own report/CSV files
SFX = f"_{os.environ['SGU_TAG']}" if os.environ.get('SGU_TAG') else ''
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = torchaudio.pipelines.MMS_FA
fa_model = fa.get_model(with_star=False).to(dev).eval(); fa_tok = fa.get_tokenizer(); fa_align = fa.get_aligner()
spk = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cuda:0' if dev == 'cuda' else 'cpu'})

_audio = {}
def audio(n):
    if n not in _audio:
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(D / 'audio' / f'{n}.mp3'), '-f', 's16le', '-ac', '1', '-ar', str(SR), '-'],
                             capture_output=True, check=True).stdout
        _audio[n] = torch.from_numpy(np.frombuffer(pcm, np.int16).astype(np.float32) / 32768)
    return _audio[n]

def piece(n, a, b): return audio(n)[int(a * SR):int(b * SR)]

def norm(w): return re.sub(r"[^a-z']", '', w.lower())

def align(n, words, a, b):
    """Forced-align the Whisper words inside [a, b]; returns [(word, start, end)] in episode seconds."""
    ws = [(w, norm(w['w'])) for w in words if w['e'] > a and w['s'] < b]
    ws = [(w, t) for w, t in ws if t and t.replace("'", '')]
    wav = piece(n, a, b).unsqueeze(0).to(dev)
    with torch.inference_mode():
        em, _ = fa_model(wav)
        spans = fa_align(em[0], fa_tok([t for _, t in ws]))
    ratio = wav.size(1) / em.size(1) / SR
    return [(w, a + s[0].start * ratio, a + s[-1].end * ratio) for (w, _), s in zip(ws, spans)]

def embed(x):
    with torch.inference_mode():
        e = spk.encode_batch(x.unsqueeze(0).to(dev)).squeeze()
    return torch.nn.functional.normalize(e, dim=0)

def host_ref(n, words, seg):
    """Host voice: 30 s starting where the first item is read (the host reads all three items)."""
    i = next((i for i in range(seg['a'], seg['b']) if norm(words[i]['w']) == 'item'), None)
    t0 = words[i]['s'] if i is not None else seg['start'] + 5  # some episodes say "number one" rather than "item"
    return embed(piece(n, t0, t0 + 30))

if __name__ == '__main__':
    source = sys.argv[1]
    rows = [r for r in csv.DictReader(open(D / f'handoffs_{source}{SFX}.csv')) if r['kind'] == 'cue' and r.get('order_ok') != '0']
    (D / 'clips').mkdir(exist_ok=True)
    out, refs = [], {}
    for r in rows:
        n = r['episode']; tx = json.loads((D / 'tx' / f'{n}.json').read_text())['words']
        seg = json.loads((D / 'seg' / f'{n}.json').read_text())
        if n not in refs: refs[n] = host_ref(n, tx, seg)
        cs, ne = float(r['cue_start']), float(r['name_end'])
        al = align(n, tx, cs - 2.0, ne + 2.0)
        # aligned spans for the cue words and the name, matched to the Whisper words by start time
        mine = [(w, s, e) for w, s, e in al if cs - 0.01 <= w['s'] <= ne]
        cue = [(w, s, e) for w, s, e in mine if norm(w['w']) in ('okay', 'ok', 'all', 'right', 'righty', 'alright', 'alrighty')]
        if not cue:  # the aligner dropped the cue words
            print(n, r['handoff'], repr(r['cue_text']), 'ALIGNMENT FAILED, skipped', flush=True); continue
        a_cue, e_cue, e_name = cue[0][1], cue[-1][2], mine[-1][2]
        # the rogue who just answered: the 6 s before the cue
        rogue = embed(piece(n, max(a_cue - 6.5, 0), a_cue - 0.5))
        x = embed(piece(n, a_cue - 0.05, e_name + 0.05))
        host_sim, rogue_sim = float(x @ refs[n]), float(x @ rogue)
        for kind, b in (('cue', e_cue), ('cue_name', e_name)):
            sf.write(str(D / 'clips' / f'{n}_h{r["handoff"]}_{kind}.wav'), piece(n, a_cue - 0.05, b + 0.05).numpy(), SR)
        r.update(cue_start=round(a_cue, 3), cue_end=round(e_cue, 3), name_end=round(e_name, 3),
                 host_sim=round(host_sim, 3), rogue_sim=round(rogue_sim, 3), host_voice=int(host_sim > rogue_sim),
                 # a spoken "okay" or "all right" is 0.2-0.8 s; anything else means the aligner slipped
                 cut_ok=int(0.15 <= e_cue - a_cue <= 0.8))
        out.append(r)
        print(n, r['handoff'], repr(r['cue_text']), f'{e_cue - a_cue:.2f}s', 'host' if r['host_voice'] else 'NOT HOST', host_sim, rogue_sim, flush=True)
    if not out: print('no clips to cut'); sys.exit(0)
    with open(D / f'clips_{source}{SFX}.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
