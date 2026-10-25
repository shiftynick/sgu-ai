# Re-transcribe a short window of an episode without VAD (handy for checking a missed word).
# Run with run_whisper.sh. usage: tx_win.py <audio> <start> <end>
import sys,os,subprocess,numpy as np
from faster_whisper import WhisperModel
src,a,b=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
pcm=subprocess.run(["ffmpeg","-v","error","-ss",str(a),"-t",str(b-a),"-i",src,"-f","s16le","-ac","1","-ar","16000","-"],capture_output=True).stdout
m=WhisperModel('large-v3-turbo',device='cuda',compute_type='float16',download_root=os.path.expanduser('~/models/whisper'))
for s in m.transcribe(np.frombuffer(pcm,np.int16).astype(np.float32)/32768,word_timestamps=True,vad_filter=False,beam_size=5)[0]:
    print(' '.join(f"{w.word.strip()}@{w.start+a:.2f}" for w in s.words))
