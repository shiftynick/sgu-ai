"""Listening test: can a person pick the fiction from how the host reads the three items? (readings from readings.py)
Serves a page on http://localhost:8766 that plays an episode's three readings in order. Press 1, 2 or 3 for the one
you think is the fiction; A/S/D replays one reading. Chance is 1 in 3. You see your score every 10 episodes.
With --filtered the audio is low-passed at 400 Hz: the words become unintelligible but pitch, rhythm and loudness
remain, so you're judging his voice and not the content (the rogues' job). Without it you hear everything.
Answers go to data/listen_items.csv.
usage: listen_items.py [--filtered] [--port 8766]"""
import csv, json, random, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
D = Path(__file__).resolve().parent.parent / 'data'
FILTERED = '--filtered' in sys.argv
PORT = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8766
OUT = D / 'listen_items.csv'
eps = {}
for r in csv.DictReader(open(D / 'readings.csv')): eps.setdefault(r['episode'], {})[r['item']] = r

def answered(who):
    if not OUT.exists(): return []
    return [r for r in csv.DictReader(open(OUT)) if r['listener'] == who]

def wav(n, item):
    r = eps[n][item]; a, b = float(r['start']), float(r['end'])
    af = ['-af', 'lowpass=f=400,lowpass=f=400,volume=2'] if FILTERED else []
    return subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-t', str(b - a), '-i', str(D / 'audio' / f'{n}.mp3'), *af,
                           '-ac', '1', '-ar', '16000', '-f', 'wav', '-'], capture_output=True, check=True).stdout

PAGE = """<!doctype html><meta charset=utf-8><title>Which is the fiction?</title>
<style>body{font:18px system-ui;max-width:680px;margin:60px auto;text-align:center}button{font-size:20px;padding:14px 24px;margin:8px}
.on{outline:3px solid #48f}#score{margin-top:30px;color:#555}</style>
<h1>Which one is the fiction?</h1>
<p>Steve reads three Science or Fiction items. One is fake. MODE</p>
<p><input id=who placeholder="your name"></p>
<p><button id=b0 onclick=play(0)>&#9654; Item 1 (A)</button><button id=b1 onclick=play(1)>&#9654; Item 2 (S)</button><button id=b2 onclick=play(2)>&#9654; Item 3 (D)</button></p>
<p><button onclick=answer(1)>Fiction is 1</button><button onclick=answer(2)>Fiction is 2</button><button onclick=answer(3)>Fiction is 3</button></p>
<div id=n></div><div id=score></div>
<script>
let cur=null, au=[0,1,2].map(()=>new Audio()), chain=true;
const who=document.getElementById('who'); who.value=localStorage.who||'';
function mark(k){[0,1,2].forEach(i=>document.getElementById('b'+i).classList.toggle('on',i==k));}
au.forEach((a,i)=>{a.onended=()=>{mark(-1); if(chain&&i<2) play(i+1, true);};});
function play(k, auto){ chain=!!auto; au.forEach(a=>a.pause()); au[k].currentTime=0; au[k].play(); mark(k);}
async function next(){cur=await (await fetch('/next?who='+encodeURIComponent(who.value))).json();
  au.forEach((a,i)=>a.src='/reading/'+cur.ep+'_'+(i+1)); document.getElementById('n').textContent=cur.done+' answered'; play(0,true);}
async function answer(g){ if(!who.value){alert('Enter your name first');return} localStorage.who=who.value;
  if(!cur){next();return}
  const r=await (await fetch('/answer',{method:'POST',body:JSON.stringify({who:who.value,ep:cur.ep,guess:g})})).json();
  if(r.summary) document.getElementById('score').textContent=r.summary; next();}
document.onkeydown=e=>{ if(e.target==who) return; const k={'1':1,'2':2,'3':3}[e.key]; if(k) answer(k);
  const p={'a':0,'s':1,'d':2}[e.key]; if(p!==undefined) play(p);};
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
        if u.path == '/':
            mode = 'The audio is muffled so you can\'t make out the words: judge his voice only.' if FILTERED else 'Judge however you like.'
            return self.send(PAGE.replace('MODE', mode), 'text/html')
        if u.path == '/next':
            who = parse_qs(u.query).get('who', [''])[0]
            seen = {r['episode'] for r in answered(who)}
            return self.send(json.dumps({'ep': random.choice([n for n in eps if n not in seen]), 'done': len(seen)}))
        if u.path.startswith('/reading/'):
            n, item = u.path[9:].split('_')
            return self.send(wav(n, item), 'audio/wav')
        self.send_error(404)
    def do_POST(self):
        a = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        truth = next(k for k, r in eps[a['ep']].items() if r['is_fiction'] == '1')
        new = not OUT.exists()
        with open(OUT, 'a', newline='') as f:
            w = csv.writer(f)
            if new: w.writerow(['time', 'listener', 'episode', 'filtered', 'guess', 'truth'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), a['who'], a['ep'], int(FILTERED), a['guess'], truth])
        mine = [m for m in answered(a['who']) if m['filtered'] == str(int(FILTERED))]
        summary = None
        if len(mine) % 10 == 0:
            right = sum(m['guess'] == m['truth'] for m in mine)
            summary = f'{right} of {len(mine)} correct ({right / len(mine):.0%}); guessing at random gets about 33%'
        self.send(json.dumps({'summary': summary}))

if __name__ == '__main__':
    print(f'item listening test on http://localhost:{PORT} ({"filtered: voice only" if FILTERED else "full audio"}); answers go to {OUT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
