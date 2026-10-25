"""Re-transcribe clips so the cuts can be checked by eye (all clips, or only the given episodes' clips).
Run with run_whisper.sh. usage: check_clips.py [episodes]"""
import glob, os, wave
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
D = Path(__file__).resolve().parent.parent / 'data'
m = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16', download_root=os.path.expanduser('~/models/whisper'))
import sys
only = set(sys.argv[1:])
for f in sorted(glob.glob(str(D / 'clips' / '*.wav'))):
    if only and os.path.basename(f).split('_')[0] not in only: continue
    w = wave.open(f); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    # pad with silence: Whisper is unreliable on sub-second audio
    a = np.concatenate([np.zeros(8000, np.float32), a, np.zeros(8000, np.float32)])
    print(f'{os.path.basename(f):24} {w.getnframes() / 16000:.2f}s |', ''.join(s.text for s in m.transcribe(a, beam_size=5, language='en')[0]).strip())
