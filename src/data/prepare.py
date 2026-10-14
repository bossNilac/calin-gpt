import hashlib
import itertools
import json
import random

import numpy as np

from Week1.day7.main import split_dataset
from src.data.acquire import cached_documents, clean_text
from src.runtime import ROOT, fingerprint, settings_from_cli
from src.tokeniser.bpe import encode_bpe, read_tokenizer, train_tokenizer


def document_batches(documents, source, config, seen, rng, size=20):
    while True:
        batch = []
        for doc in itertools.islice(documents, size):
            text = clean_text(doc['text'])
            key = hashlib.sha256(' '.join(text.split()).casefold().encode()).hexdigest()
            if not text or len(text) > config['max_document_chars'] or key in seen:
                continue
            seen.add(key)
            batch.append(dict(doc, text=text, sha256=key, source=source))
        if not batch:
            # A batch of rejected documents is not necessarily end-of-stream.
            try:
                first = next(documents)
            except StopIteration:
                return
            documents = itertools.chain([first], documents)
            continue
        rng.shuffle(batch)
        train, _, val = split_dataset(batch, 1 - config['validation_ratio'], 0,
                                      config['validation_ratio'])
        yield train, val


def prepare(settings):
    config = settings['data']
    output = ROOT / config['processed_dir']
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / config['tokenizer_path']
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    debug = settings['runtime']['device'] == 'cpu'
    sources = ['local'] if debug else list(config['mixture'])
    rng = random.Random(config['seed'])
    seen = set()
    batches, initial = {}, {}
    for source in sources:
        if debug:
            text = (ROOT / 'data/fixtures/english.txt').read_text(encoding='utf-8')
            docs = iter({'text': part, 'origin': 'data/fixtures/english.txt'} for part in text.split('\n---\n'))
        else:
            docs = iter(cached_documents(source, config))
        batches[source] = document_batches(docs, source, config, seen, rng)
        initial[source] = next(batches[source], ([], []))
    if tokenizer_path.exists():
        vocab, merges, eod = read_tokenizer(tokenizer_path)
    else:
        sample = []
        for source in sources:
            quota = int(config['tokenizer_train_chars'] * (1 if debug else config['mixture'][source]))
            sample.append('\n\n'.join(doc['text'] for doc in initial[source][0])[:quota])
        vocab, merges, eod = train_tokenizer('\n\n'.join(sample), settings['model']['vocab_size'], tokenizer_path)
    if len(vocab) != settings['model']['vocab_size'] or eod is None:
        raise ValueError('Tokenizer vocabulary/EOD differs from config; use a new artifact path')
    dtype = '<u2' if len(vocab) <= 65536 else '<u4'
    records = {'train': [], 'val': []}
    counts = {}
    manifest = []
    for source in sources:
        counts[source] = {'train': 0, 'val': 0}
        total = config['target_tokens'] * config['mixture'].get(source, 1)
        limits = {'train': int(total * (1 - config['validation_ratio'])),
                  'val': int(total * config['validation_ratio'])}
        for train, val in itertools.chain([initial[source]], batches[source]):
            for split, docs in [('train', train), ('val', val)]:
                for doc in docs:
                    if not debug and counts[source][split] >= limits[split]:
                        break
                    ids = encode_bpe(doc['text'], merges) + [eod]
                    if not debug:
                        ids = ids[:limits[split] - counts[source][split]]
                        ids[-1] = eod
                    records[split].append(np.asarray(ids, dtype=dtype))
                    counts[source][split] += len(ids)
                    manifest.append({k: v for k, v in dict(doc, split=split, tokens=len(ids)).items() if k != 'text'})
            if debug or all(counts[source][s] >= limits[s] for s in limits):
                break
        if not debug and any(counts[source][s] < limits[s] for s in limits):
            print(f'Source shortfall: {source}: got {counts[source]}, wanted {limits}; no substitution')
    streams = {}
    for split, arrays in records.items():
        rng.shuffle(arrays)
        path = output / f'{split}.bin'
        with path.open('wb') as f:
            for array in arrays:
                array.tofile(f)
        streams[split] = {'tokens': sum(map(len, arrays)), 'sha256': fingerprint(path)}
        if streams[split]['tokens'] <= settings['model']['context_length']:
            raise ValueError(f'Not enough {split} tokens')
    (output / 'documents.json').write_text(json.dumps(manifest, indent=2))
    metadata = {'version': 1, 'dtype': dtype, 'vocab_size': len(vocab), 'eod_id': eod,
                'tokenizer_sha256': fingerprint(tokenizer_path), 'seed': config['seed'],
                'sources': counts, 'streams': streams, 'data_config': config,
                'documents_sha256': fingerprint(output / 'documents.json')}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Prepared:', config['processed_dir'], 'source token counts:', counts)
    return metadata


if __name__ == '__main__':
    prepare(settings_from_cli())
