"""Listening test: can a person hear whether the guess before an "Okay" was right?
Serves a page on http://localhost:8765 that plays random clips from data/dataset.csv, half after a right guess
and half after a wrong one. Press R (sounds like the guess was right) or W (wrong), Space to replay.
You only see your score every 20 clips, so you can't learn from per-clip feedback.
Answers go to data/listen.csv, one row per clip, with the listener's name.
usage: listen.py [--cue-only] [--port 8765]"""
import csv, json, random, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
D = Path(__file__).resolve().parent.parent / 'data'
CUE_ONLY = '--cue-only' in sys.argv
PORT = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8765
rows = list(csv.DictReader(open(D / 'dataset.csv')))
by_label = {l: [r for r in rows if r['label_prev_correct'] == l] for l in '01'}
OUT = D / 'listen.csv'

def answered(who):
    if not OUT.exists(): return []
    return [r for r in csv.DictReader(open(OUT)) if r['listener'] == who]

PAGE = """<!doctype html><meta charset=utf-8><title>Okay, Jay</title>
<style>body{font:18px system-ui;max-width:640px;margin:60px auto;text-align:center}button{font-size:20px;padding:14px 28px;margin:10px}
#score{margin-top:30px;color:#555}</style>
<h1>Was the guess right?</h1>
<p>Steve has just heard a rogue's guess. Listen to how he hands off to the next rogue.</p>
<p><input id=who placeholder="your name"></p>
<p><button onclick=play()>&#9654; Replay (Space)</button></p>
<p><button onclick="answer(1)">Right (R)</button><button onclick="answer(0)">Wrong (W)</button></p>
<div id=n></div><div id=score></div>
<script>
let cur=null, a=new Audio();
const who=document.getElementById('who'); who.value=localStorage.who||'';
async function next(){cur=await (await fetch('/next?who='+encodeURIComponent(who.value))).json(); a.src='/clip/'+cur.id; play();
  document.getElementById('n').textContent=cur.done+' answered';}
function play(){a.currentTime=0; a.play();}
async function answer(g){ if(!who.value){alert('Enter your name first');return} localStorage.who=who.value;
  if(!cur){next();return}
  const r=await (await fetch('/answer',{method:'POST',body:JSON.stringify({who:who.value,id:cur.id,guess:g})})).json();
  if(r.summary) document.getElementById('score').textContent=r.summary; next();}
document.onkeydown=e=>{ if(e.target==who) return; if(e.key=='r')answer(1); if(e.key=='w')answer(0); if(e.key==' '){e.preventDefault();play();}};
who.onchange=()=>{localStorage.who=who.value; next();}; if(who.value) next();
</script>"""

class H(BaseHTTPRequestHandler):
    def send(self, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b))); self.end_headers()
        self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/': return self.send(PAGE, 'text/html')
        if u.path == '/next':
            who = parse_qs(u.query).get('who', [''])[0]
            seen = {r['id'] for r in answered(who)}
            pool = [r for r in by_label[random.choice('01')] if f"{r['episode']}_{r['handoff']}" not in seen]
            r = random.choice(pool)
            return self.send(json.dumps({'id': f"{r['episode']}_{r['handoff']}", 'done': len(seen)}))
        if u.path.startswith('/clip/'):
            n, h = u.path[6:].split('_')
            r = next(r for r in rows if r['episode'] == n and r['handoff'] == h)
            return self.send((D / r['clip' if CUE_ONLY else 'clip_with_name']).read_bytes(), 'audio/wav')
        self.send_error(404)
    def do_POST(self):
        a = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        n, h = a['id'].split('_')
        r = next(r for r in rows if r['episode'] == n and r['handoff'] == h)
        new = not OUT.exists()
        with open(OUT, 'a', newline='') as f:
            w = csv.writer(f)
            if new: w.writerow(['time', 'listener', 'id', 'clip', 'guess', 'truth'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), a['who'], a['id'], 'cue' if CUE_ONLY else 'cue+name', a['guess'], r['label_prev_correct']])
        mine = answered(a['who'])
        summary = None
        if len(mine) % 20 == 0:
            right = sum(m['guess'] == m['truth'] for m in mine)
            summary = f'{right} of {len(mine)} correct ({right / len(mine):.0%}); guessing at random gets about 50%'
        self.send(json.dumps({'summary': summary}))

if __name__ == '__main__':
    print(f'listening test on http://localhost:{PORT} ({"cue only" if CUE_ONLY else "cue + name"}); answers go to {OUT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
