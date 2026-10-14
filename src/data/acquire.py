import json
import re
import time
from urllib.request import Request, urlopen

from src.runtime import ROOT, fingerprint


def fetch(url):
    request = Request(url, headers={'User-Agent': 'Calin-GPT/1.0 (bounded research dataset preparation)'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\ufeff', '')
    if '\ufffd' in text or '\x00' in text:
        return ''
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text if len(text) >= 80 and sum(c.isalpha() for c in text) > len(text) * 0.4 else ''


def strip_gutenberg(text):
    start = re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*', text, re.I)
    end = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*', text, re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError('Gutenberg header/footer markers missing')
    return text[start.end():end.start()]


def wiki_articles(rows):
    lines = []
    for row in rows:
        text = row['text']
        if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and lines:
            if any(line.strip() for line in lines):
                yield {'text': '\n'.join(lines)}
            lines = []
        lines.append(text)
    if any(line.strip() for line in lines):
        yield {'text': '\n'.join(lines)}


def source_documents(source, config):
    cache = ROOT / config['cache_dir']
    cache.mkdir(parents=True, exist_ok=True)
    if source == 'gutenberg':
        for book in config['gutenberg_ids']:
            digits = '/'.join(str(book)[:-1]) if book >= 10 else '0'
            url = f"{config['gutenberg_mirror']}/{digits}/{book}/{book}-0.txt"
            path = cache / f'pg{book}.txt'
            if not path.exists():
                time.sleep(config['gutenberg_delay'])
                content = fetch(url)
                content.decode('utf-8-sig')
                path.write_bytes(content)
            yield {'text': strip_gutenberg(path.read_text(encoding='utf-8-sig')),
                   'origin': url, 'book_id': book, 'raw_sha256': fingerprint(path)}
        return

    from datasets import load_dataset
    spec = config['hf'][source]
    revision_file = cache / f'{source}-revision.json'
    if not revision_file.exists():
        revision_file.write_text(json.dumps({'repo': spec['repo'], 'revision': spec['revision']}))
    revision = json.loads(revision_file.read_text())
    if revision['repo'] != spec['repo'] or revision['revision'] != spec['revision']:
        raise ValueError('Cached source repository differs from configuration')
    rows = load_dataset(spec['repo'], name=spec['config'], revision=revision['revision'],
                        split='train', streaming=True, cache_dir=str(cache / 'hf'))
    documents = wiki_articles(rows) if source == 'wikitext' else rows
    for index, row in enumerate(documents):
        yield {'text': row['text'], 'origin': row.get('url', spec['repo']),
               'source_id': row.get('id', index), 'revision': revision['revision']}


def cached_documents(source, config):
    path = ROOT / config['cache_dir'] / f'{source}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = (dict(ids=config['gutenberg_ids'], mirror=config['gutenberg_mirror'])
            if source == 'gutenberg' else config['hf'][source])
    identity_path = path.with_suffix('.source.json')
    if identity_path.exists() and json.loads(identity_path.read_text()) != spec:
        raise ValueError('Source configuration changed; choose a new cache directory')
    identity_path.write_text(json.dumps(spec))
    count = 0
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                count += 1
                yield json.loads(line)
    if count >= config['max_source_documents'] or path.with_suffix('.done').exists():
        return
    # Only reconnect after the cached prefix has been consumed.
    with path.open('a', encoding='utf-8') as f:
        for index, doc in enumerate(source_documents(source, config)):
            if index < count:
                continue
            if index >= config['max_source_documents']:
                return
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            f.flush()
            yield doc
    path.with_suffix('.done').touch()
