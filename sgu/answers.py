"""Build data/episodes.json from the RSS feed and scrape each episode page's Science or Fiction answer key.
usage: answers.py <episode numbers...>"""
import html, json, re, subprocess, sys
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'

def feed():
    x = (D / 'feed.xml').read_text()
    eps = {}
    for it in x.split('<item>')[1:]:
        m = re.search(r'<title>(?:The )?Skeptics.? Guide #(\d+) - ([^<]+)</title>', it)
        if not m: continue
        eps[int(m[1])] = {'n': int(m[1]), 'date': m[2].strip(), 'mp3': re.search(r'enclosure url="([^"]+)', it)[1],
                          'desc': html.unescape(re.search(r'<description>(.*?)</description>', it, re.S)[1])}
    return eps

def page_text(n):
    p = D / 'pages' / f'{n}.html'
    if not p.exists():
        subprocess.run(['curl', '-sL', '-m', '60', '-o', str(p), f'https://www.theskepticsguide.org/podcasts/episode-{n}'], check=True)
    t = html.unescape(re.sub(r'<[^>]+>', '\n', p.read_text(errors='replace')))
    return [l.strip() for l in t.splitlines() if l.strip()]

def key(n):
    lines = page_text(n)
    try: i = lines.index('[ Show Answers ]')
    except ValueError: return None
    items = []
    j = i + 1
    while j + 1 < len(lines) and re.fullmatch(r'Item #\d+', lines[j]):
        num, label = int(lines[j][6:]), lines[j + 1]; j += 2
        # early episodes list labels with no item text
        text = '' if j >= len(lines) or re.fullmatch(r'Item #\d+|Skeptical Quote of the Week\.', lines[j]) else lines[j]
        if text: j += 1
        while j < len(lines) and re.match(r'https?://', lines[j]): j += 1  # source links under an item
        items.append({'num': num, 'label': label, 'text': text})
    return items

if __name__ == '__main__':
    eps = feed()
    out = D / 'episodes.json'
    db = json.loads(out.read_text()) if out.exists() else {}
    for n in map(int, sys.argv[1:]):
        e = eps[n]; items = key(n)
        e['items'] = items
        fic = [it['num'] for it in items or [] if it['label'].lower() == 'fiction']
        has_text = bool(items) and all(it['text'] for it in items)
        # without item text the page's labels are placeholders; the fiction then comes from the host's reveal
        e['fiction'] = fic[0] if has_text and len(fic) == 1 else None
        e['key_source'] = 'site' if e['fiction'] else 'transcript'
        e['ok'] = bool(items) and len(items) == 3 and (e['fiction'] is not None or not has_text)
        db[str(n)] = e
        print(n, e['date'], 'items', len(items or []), 'fiction', e['fiction'], e['key_source'], 'OK' if e['ok'] else 'SKIP', '|', e['desc'][:90])
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(db, indent=1)); tmp.replace(out)  # atomic: other steps read it concurrently
