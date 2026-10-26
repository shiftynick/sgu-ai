"""Run the whole pipeline over the archive in batches, in the background, resumable.

  python3 -I sgu/run.py start [--limit N] [--batch 25]   start (or resume) in the background
  python3 -I sgu/run.py stop [--now]                     finish the current episode, then stop (--now: kill)
  python3 -I sgu/run.py status                           progress, dataset size, recent log

Episodes are processed in a fixed random order, so a partial run is still a spread across the archive.
Each episode is marked in data/run/done/<n> once it has gone all the way through (or been skipped, with
the reason); a restart picks up the next unmarked episodes. While batch k is in the post stage
(Haiku, checks, cutting), batch k+1 is already downloading and transcribing.
"""
import json, os, random, re, shutil, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
D, S = ROOT / 'data', ROOT / 'sgu'
RUN = D / 'run'
PY, WHISPER = str(ROOT / '.venv' / 'bin' / 'python'), str(S / 'run_whisper.sh')
SKIP_DESC = re.compile(r'\blive\b|extravaganza|year in review', re.I)

def log(*a):
    with open(RUN / 'run.log', 'a') as f: print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, file=f, flush=True)

def sh(cmd, env=None, out=None):
    """Run a pipeline step; its output goes to the step log."""
    with open(out or RUN / 'steps.log', 'a') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode

def stopping(): return (RUN / 'STOP').exists()

def mark(n, status):
    (RUN / 'done').mkdir(parents=True, exist_ok=True); (RUN / 'done' / str(n)).write_text(status)

def done(): return {int(p.name) for p in (RUN / 'done').glob('*')} if (RUN / 'done').exists() else set()

def order():
    sys.path.insert(0, str(S)); import answers
    nums = sorted(answers.feed()); random.Random(2026).shuffle(nums)
    return nums

def eps(): return json.loads((D / 'episodes.json').read_text())

def front(batch):
    """Answer keys, downloads, transcription and segmenting for one batch. Returns the episodes ready for post."""
    sh(['python3', '-I', str(S / 'answers.py'), *map(str, batch)])
    db, ready = eps(), []
    for n in batch:
        e = db.get(str(n))
        if not e or not e['ok']: mark(n, 'skipped: answer key unusable')
        elif SKIP_DESC.search(e['desc']): mark(n, 'skipped: live or special episode')
        else: ready.append(n)
    if not ready: return []
    sh(['python3', '-I', str(S / 'download.py'), *map(str, ready)])
    for n in ready:
        mp3 = D / 'audio' / f'{n}.mp3'
        if not mp3.exists(): mark(n, 'skipped: download failed')
    ready = [n for n in ready if (D / 'audio' / f'{n}.mp3').exists()]
    sh([WHISPER, str(S / 'transcribe.py'), *map(str, ready)])
    if not stopping():
        for n in ready:
            if not (D / 'tx' / f'{n}.json').exists(): mark(n, 'skipped: unreadable audio')
    ready = [n for n in ready if (D / 'tx' / f'{n}.json').exists()]  # a stop request leaves the rest for later
    out = []
    for n in ready:
        if not (D / 'seg' / f'{n}.json').exists(): sh(['python3', '-I', str(S / 'segment.py'), str(n)])
        if not (D / 'seg' / f'{n}.json').exists() and json.loads((D / 'tx' / f'{n}.json').read_text())['start'] > 0 and not stopping():
            # early episodes run the segment near the start: transcribe the whole thing and look again
            sh([WHISPER, str(S / 'transcribe.py'), '--full', str(n)]); sh(['python3', '-I', str(S / 'segment.py'), str(n)])
        if (D / 'seg' / f'{n}.json').exists(): out.append(n)
        elif (D / 'tx' / f'{n}.json').exists() and json.loads((D / 'tx' / f'{n}.json').read_text())['start'] == 0:
            mark(n, 'skipped: no Science or Fiction segment')
    return out

def post(tag, batch):
    """Everything after segmenting, for one batch (runs as its own process)."""
    nums, env = list(map(str, batch)), {'SGU_TAG': tag}
    for b in ('haiku', 'haiku_b'): sh([PY, '-I', str(S / 'extract.py'), b, *nums])
    def turns(b, n):
        p = D / 'extract' / b / f'{n}.json'
        return [(t['rogue'], t['pick']) for t in json.loads(p.read_text())['turns']] if p.exists() else None
    disputed = [n for n in nums if turns('haiku', n) != turns('haiku_b', n)]
    if disputed: sh([PY, '-I', str(S / 'extract.py'), 'haiku_tiebreak', *disputed])
    sh(['python3', '-I', str(S / 'consensus.py'), 'haiku', 'haiku_b', *nums], env)
    sh([PY, '-I', str(S / 'recap.py'), *nums], env)
    sh(['python3', '-I', str(S / 'handoffs.py'), 'consensus', *nums], env)
    sh([PY, '-I', str(S / 'cut.py'), 'consensus'], env)
    sh([WHISPER, str(S / 'check_clips.py'), *nums], out=D / 'check_clips.log')
    sh(['python3', '-I', str(S / 'clip_text.py'), 'consensus'], env)
    sh(['python3', '-I', str(S / 'dataset.py')])
    for n in nums:
        if not (D / 'extract' / 'haiku' / f'{n}.json').exists(): continue  # extraction failed: retry next start
        mark(n, 'processed')
    log(f'batch {tag} post done:', ' '.join(nums))

def loop(limit, size):
    (RUN / 'STOP').unlink(missing_ok=True)
    todo = [n for n in order() if n not in done()][:limit]
    log(f'start: {len(todo)} episodes to do, batches of {size}')
    pending = None
    for k in range(0, len(todo), size):
        if stopping(): break
        batch = todo[k:k + size]; tag = time.strftime('%Y%m%d%H%M%S')
        log(f'batch {tag}: front stage for', ' '.join(map(str, batch)))
        ready = front(batch)
        if pending: pending.wait()
        if ready:
            pending = subprocess.Popen(['python3', '-I', str(S / 'run.py'), '_post', tag, *map(str, ready)], cwd=ROOT)
    if pending: pending.wait()
    log('stopped on request' if stopping() else 'finished')
    (RUN / 'pid').unlink(missing_ok=True)

def running():
    try:
        pid = int((RUN / 'pid').read_text()); os.kill(pid, 0); return pid
    except (FileNotFoundError, ValueError, ProcessLookupError): return None

def status():
    import csv, glob
    pid = running(); d = {}
    for p in (RUN / 'done').glob('*') if (RUN / 'done').exists() else []:
        r = p.read_text(); d[r] = d.get(r, 0) + 1
    total = len(order())
    print(f"runner: {'running (pid %d)' % pid if pid else 'not running'}{', stop requested' if stopping() else ''}")
    print(f"episodes: {sum(d.values())} of {total} done ->", ', '.join(f'{v} {k}' for k, v in sorted(d.items(), key=lambda x: -x[1])))
    print(f"transcribed: {len(list((D / 'tx').glob('*.json')))}, segments: {len(list((D / 'seg').glob('*.json')))}, "
          f"extracted: {len(list((D / 'extract' / 'consensus').glob('*.json')))} agreed")
    if (D / 'dataset.csv').exists():
        rows = list(csv.DictReader(open(D / 'dataset.csv'))); pos = sum(r['label_prev_correct'] == '1' for r in rows)
        print(f"dataset: {len(rows)} clips from {len({r['episode'] for r in rows})} episodes ({pos} after correct, {len(rows) - pos} after wrong)")
    if (RUN / 'run.log').exists():
        print('recent log:'); print(''.join(open(RUN / 'run.log').readlines()[-5:]), end='')

if __name__ == '__main__':
    RUN.mkdir(parents=True, exist_ok=True)
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else 'status'), sys.argv[2:]
    if cmd == 'start':
        if running(): sys.exit(f'already running (pid {running()})')
        limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 10 ** 6
        size = int(args[args.index('--batch') + 1]) if '--batch' in args else 25
        # its own systemd scope, so closing the terminal or app that started it doesn't kill it with its process group
        # (systemd-run --scope execs the command, so the pid stays the same)
        scope = ['systemd-run', '--user', '--scope', '--quiet', '--collect'] if shutil.which('systemd-run') else []
        p = subprocess.Popen([*scope, 'python3', '-I', str(S / 'run.py'), '_loop', str(limit), str(size)], cwd=ROOT,
                             stdout=open(RUN / 'runner.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
        (RUN / 'pid').write_text(str(p.pid)); print(f'started (pid {p.pid}); log: {RUN / "run.log"}')
    elif cmd == 'stop':
        pid = running()
        if not pid: sys.exit('not running')
        if '--now' in args:
            os.killpg(pid, signal.SIGTERM); (RUN / 'pid').unlink(missing_ok=True); print('killed; unfinished episodes will be redone on the next start')
        else:
            (RUN / 'STOP').touch(); print('stop requested: it finishes the episode it is transcribing and the batch already in post, then exits')
    elif cmd == 'status': status()
    elif cmd == '_loop': loop(int(args[0]), int(args[1]))
    elif cmd == '_post': post(args[0], args[1:])
    else: sys.exit(__doc__)
