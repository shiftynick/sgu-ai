"""Download each episode and transcribe its second half with word timestamps -> data/tx/<n>.json.
With --full, transcribe the whole episode instead (replacing a second-half transcript): early episodes ran
Science or Fiction near the start. Run with the AI Watch audio venv (see run_whisper.sh).
usage: transcribe.py [--full] <episode numbers...>"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
D = Path(__file__).resolve().parent.parent / 'data'
eps = json.loads((D / 'episodes.json').read_text())
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16',
                     download_root=os.path.expanduser('~/models/whisper'))
full = '--full' in sys.argv
for n in [a for a in sys.argv[1:] if a != '--full']:
    if (D / 'run' / 'STOP').exists(): print('stop requested', flush=True); break
    out = D / 'tx' / f'{n}.json'
    if out.exists() and (not full or json.loads(out.read_text())['start'] == 0): continue
    mp3 = D / 'audio' / f'{n}.mp3'
    while not mp3.exists() and mp3.with_suffix('.part').exists():  # download.py is still fetching it
        time.sleep(5)
    if not mp3.exists():
        subprocess.run(['curl', '-sL', '-m', '600', '-o', str(mp3), eps[n]['mp3']], check=True)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(mp3)],
                           capture_output=True, text=True).stdout.strip()
    try: dur = float(probe)
    except ValueError:
        print(n, 'UNREADABLE AUDIO, skipped (delete the mp3 to re-download)', flush=True); continue
    start = 0.0 if full else dur / 2
    # decode ourselves: faster-whisper's own file decoding breaks with this PyAV version
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(mp3), '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    segs, _ = model.transcribe(np.frombuffer(pcm, np.int16).astype(np.float32) / 32768, word_timestamps=True, vad_filter=True,
                               beam_size=1, initial_prompt='The Skeptics Guide to the Universe. Steve Novella, Bob, Jay, Cara, Evan. Science or Fiction.')
    words = [{'w': w.word, 's': round(w.start + start, 2), 'e': round(w.end + start, 2), 'p': round(w.probability, 3)}
             for s in segs for w in s.words]
    out.write_text(json.dumps({'n': int(n), 'start': start, 'duration': dur, 'words': words}))
    print(n, f'{dur/60:.0f} min', len(words), 'words', flush=True)
