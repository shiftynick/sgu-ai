"""Find the Science or Fiction segment in each transcript and write data/seg/<n>.json:
word range plus a readable transcript with a [seconds] marker at each pause. usage: segment.py <episode numbers...>"""
import json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
MAX_LEN = 35 * 60

def find(words):
    text, offs = '', []
    for w in words:
        offs.append(len(text)); text += w['w']
    def word_at(c):  # index of the word containing character offset c
        return max(i for i, o in enumerate(offs) if o <= c)
    intro = re.search(r"time for science (or|and) fiction|time for science fiction|on to science (or|and) fiction", text, re.I)
    if not intro:  # fall back to the first mention followed by an item being read
        intro = next((m for m in re.finditer(r'science (or|and) fiction', text, re.I)
                      if re.search(r'item (number )?(one|1)\b', text[m.end():m.end() + 2000], re.I)), None)
    if not intro: return None
    a = word_at(intro.start())
    t0 = words[a]['s']
    end = re.compile(r"skeptical quote|until next week|that's (going to be )?(the|our) show", re.I).search(text, intro.end())
    b = word_at(end.start()) if end else len(words) - 1
    while words[b]['s'] - t0 > MAX_LEN: b -= 1
    return a, b

def render(words):
    out, last = [], None
    for w in words:
        if last is None or w['s'] - last > 0.8: out.append(f"\n[{w['s']:.1f}]")
        out.append(w['w']); last = w['e']
    return ''.join(out).strip()

if __name__ == '__main__':
    (D / 'seg').mkdir(exist_ok=True)
    for n in sys.argv[1:]:
        tx = json.loads((D / 'tx' / f'{n}.json').read_text())
        r = find(tx['words'])
        if not r: print(n, 'NO SEGMENT FOUND'); continue
        a, b = r; ws = tx['words'][a:b + 1]
        (D / 'seg' / f'{n}.json').write_text(json.dumps({'n': int(n), 'a': a, 'b': b, 'start': ws[0]['s'], 'end': ws[-1]['e'], 'text': render(ws)}))
        print(n, f"{ws[0]['s']/60:.1f}-{ws[-1]['e']/60:.1f} min", len(ws), 'words')
