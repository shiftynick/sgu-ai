"""Ask an LLM which item each rogue picked, and in what order, for each episode's segment.
Writes data/extract/<backend>/<n>.json as {host, turns: [{rogue, pos, pick, quote, start}]}.
haiku runs through the Claude CLI (subscription); haiku_api uses ANTHROPIC_API_KEY.
usage: extract.py <qwen|haiku|haiku_api|decisions> <episode numbers...>
API keys come from the environment or the .env file next to this folder (OPENAI_API_KEY, ANTHROPIC_API_KEY)."""
import json, os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'
if (ROOT / '.env').exists():
    for line in (ROOT / '.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())

# rogues by era: Perry died in Aug 2007, Rebecca left in late 2014, Cara joined in early 2015
ERAS = {'Bob': (2005, 2100), 'Jay': (2005, 2100), 'Evan': (2005, 2100), 'Perry': (2005, 2007.9),
        'Rebecca': (2005, 2014.99), 'Cara': (2014.9, 2100)}

def year(ep):
    m = re.search(r'([A-Za-z]{3})[a-z]*\.?\s+(\d+)\w*,?\s+(\d{4})', ep['date'])
    months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
    return int(m[3]) + months.index(m[1][:3].title()) / 12 if m else 2010

def roster(ep):
    """Rogues on the show at the time plus guests named in the feed description."""
    y = year(ep)
    core = [r for r, (a, b) in ERAS.items() if a <= y <= b]
    guests = re.findall(r'(?:Guest Rogue|Special Guest Rogue|Guest Rogues?)s?:\s*([^;]+)', ep['desc'])
    names = [g.strip() for gs in guests for g in re.split(r',| and ', gs) if g.strip()]
    # "Dr. Ray Greek - The Science of Animal Models" -> "Ray Greek"
    names = [re.sub(r'^(Dr|Prof)\.?\s+', '', g.split(' - ')[0].split(':')[0]).strip() for g in names]
    return core + [n for n in names if n.split()[0] not in core]


INSTRUCTIONS = """You are labeling the "Science or Fiction" segment of The Skeptics' Guide to the Universe podcast.
The host (usually Steve Novella) reads three news items. Exactly one is fiction. The panelists ("rogues") then take turns saying which item they think is the fiction, and afterwards the host reveals the answers.
The transcript below is automatic speech recognition with no speaker labels. Names may be misspelled (for example "Kara" is Cara, "Jeff" may be Jay). Numbers in [brackets] are seconds from the start of the episode.
Work out, for each panelist who made a guess, in the order they guessed:
- rogue: their name, from this list: {roster}
- pick: the item number (1, 2 or 3) they finally committed to as the fiction before the next panelist's turn
- quote: a short verbatim quote from the transcript where they state the pick
- start: the [bracket] time nearest the start of their turn
The host's recap during the reveal (e.g. "Bob and Jay think this one is the fiction") is the most reliable evidence; use it to check your answer. Do not include the host as a panelist. Also say who hosted the segment, and which item number the host revealed as the fiction (fiction_revealed). If the segment does not follow this format (no guesses, or no clear reveal), return an empty turns list and fiction_revealed 0.

ITEMS
{items}

TRANSCRIPT
{text}"""

def build_prompt(ep, seg):
    items = '\n'.join(f"Item #{it['num']}: {it['text'] or '(text not available; read it from the transcript)'}" for it in ep['items'])
    return INSTRUCTIONS.format(roster=', '.join(roster(ep)), items=items, text=seg['text'])

def schema(names):
    return {'type': 'object', 'additionalProperties': False, 'required': ['host', 'fiction_revealed', 'turns'], 'properties': {
        'host': {'type': 'string'}, 'fiction_revealed': {'type': 'integer', 'enum': [0, 1, 2, 3]},
        'turns': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                  'required': ['rogue', 'pick', 'quote', 'start'], 'properties': {
                      'rogue': {'type': 'string', 'enum': names}, 'pick': {'type': 'integer', 'enum': [1, 2, 3]},
                      'quote': {'type': 'string'}, 'start': {'type': 'number'}}}}}}

def normalize(out):
    for i, t in enumerate(out['turns']): t['pos'] = i + 1
    return out

def run_haiku(ep, seg):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model='claude-haiku-5-5', max_tokens=8000,
        messages=[{'role': 'user', 'content': build_prompt(ep, seg)}],
        output_config={'format': {'type': 'json_schema', 'schema': schema(roster(ep))}})
    if r.stop_reason == 'refusal': raise RuntimeError(f'refusal: {r.stop_details}')
    text = next(b.text for b in r.content if b.type == 'text')
    return normalize(json.loads(text)), {'in': r.usage.input_tokens, 'out': r.usage.output_tokens}

def run_qwen(ep, seg, think=False):
    from openai import OpenAI
    key = Path('~/.local/state/omarchy/local-ai/gateway.key').expanduser().read_text().strip()
    c = OpenAI(base_url='http://127.0.0.1:12434/v1', api_key=key, timeout=3600)
    model = c.models.list().data[0].id
    r = c.chat.completions.create(model=model, max_tokens=40000 if think else 4000, temperature=0,
        messages=[{'role': 'user', 'content': build_prompt(ep, seg)}],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'sof', 'schema': schema(roster(ep))}},
        extra_body={'chat_template_kwargs': {'enable_thinking': think}})
    if r.choices[0].finish_reason != 'stop': raise RuntimeError(f'qwen stopped: {r.choices[0].finish_reason}')
    text = re.sub(r'(?s)^.*</think>', '', r.choices[0].message.content).strip()
    return normalize(json.loads(text)), {'model': model, 'think': think}


def run_claude_cli(ep, seg, model='claude-haiku-5-5'):
    """Haiku through the Claude Code CLI in headless mode, so it bills to the Claude subscription instead of an API key."""
    import subprocess, tempfile
    cmd = ['claude', '-p', '--model', model, '--tools', '', '--no-session-persistence', '--output-format', 'json',
           '--system-prompt', 'You extract structured data from podcast transcripts. Answer only with the requested JSON.',
           '--json-schema', json.dumps(schema(roster(ep)))]
    # run from an empty folder so no project CLAUDE.md is picked up
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, input=build_prompt(ep, seg), capture_output=True, text=True, cwd=cwd, timeout=900)
    if r.returncode: raise RuntimeError(f'claude -p failed: {r.stderr[-2000:] or r.stdout[-2000:]}')
    res = json.loads(r.stdout)
    if res.get('is_error'): raise RuntimeError(f"claude -p error: {res.get('result')}")
    out = res.get('structured_output') or json.loads(res['result'])
    return normalize(out), {'model': model, 'cost_usd': res.get('total_cost_usd'), 'usage': res.get('usage')}


def run_decisions(ep, seg):
    """One request per episode; per-rogue pick and order questions. The Decisions API only chooses, so no quotes."""
    from openai import OpenAI
    names = roster(ep)
    items = [{'value': str(it['num']), 'description': f"Item #{it['num']}: {it['text']}"} for it in ep['items']]
    none = {'value': 'none', 'description': 'This person did not make a guess in this segment (absent, or was the host).'}
    qs = []
    for n in names:
        qs.append({'type': 'choice', 'name': f'pick_{n}', 'choices': items + [none],
                   'instructions': f'Which item did {n} finally commit to as the fiction, before the next panelist took their turn?'})
        qs.append({'type': 'choice', 'name': f'pos_{n}', 'choices': [{'value': str(i), 'description': f'{n} was panelist number {i} to guess.'} for i in range(1, 8)] + [none],
                   'instructions': f'In what order did {n} give their guess? 1 means the first panelist to guess.'})
    qs.append({'type': 'choice', 'name': 'host', 'instructions': 'Who hosted the segment (read the items and revealed the answers)?',
               'choices': [{'value': 'Steve', 'description': 'Steve Novella'}] + [{'value': n, 'description': n} for n in names]})
    prompt = build_prompt(ep, seg).split('ITEMS')[0].split('Work out,')[0] + '\nTRANSCRIPT\n' + seg['text']
    r = OpenAI().decisions.create(model='gpt-6-luna', input=prompt, questions=qs)
    a = {x.name: x for x in r.answers}
    turns = []
    for n in names:
        p, q = a.get(f'pick_{n}'), a.get(f'pos_{n}')
        if p and p.type == 'choice' and p.choice != 'none':
            pos = int(q.choice) if q and q.type == 'choice' and q.choice != 'none' else 99
            turns.append({'rogue': n, 'pick': int(p.choice), 'quote': '', 'start': None, 'pos': pos, 'confidence': p.confidence})
    turns.sort(key=lambda t: t['pos'])
    host = a['host'].choice if a['host'].type == 'choice' else None
    return {'host': host, 'turns': turns}, {'usage': getattr(r, 'usage', None) and r.usage.model_dump()}

if __name__ == '__main__':
    backend, nums = sys.argv[1], sys.argv[2:]
    eps = json.loads((D / 'episodes.json').read_text())
    (D / 'extract' / backend).mkdir(parents=True, exist_ok=True)
    fn = {'haiku': run_claude_cli, 'haiku_b': run_claude_cli, 'haiku_c': run_claude_cli, 'haiku_tiebreak': run_claude_cli, 'haiku_api': run_haiku, 'qwen': run_qwen, 'qwen_think': lambda e, s: run_qwen(e, s, True), 'decisions': run_decisions}[backend]
    def one(n):
        out = D / 'extract' / backend / f'{n}.json'
        if out.exists(): return
        seg_p = D / 'seg' / f'{n}.json'
        if not seg_p.exists(): print(backend, n, 'no segment, skipped', flush=True); return
        try:
            res, usage = fn(eps[n], json.loads(seg_p.read_text()))
        except Exception as e:
            print(backend, n, 'FAILED', str(e)[:300], flush=True); return
        res['usage'] = usage
        out.write_text(json.dumps(res, indent=1))
        print(backend, n, res['host'], 'fiction', res.get('fiction_revealed'), [(t['rogue'], t['pick']) for t in res['turns']], flush=True)
    # API backends run a few at a time; the local Qwen server handles one request at a time
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1 if backend.startswith('qwen') else 4) as ex:
        list(ex.map(one, nums))
