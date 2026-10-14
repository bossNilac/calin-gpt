import json

import numpy as np

from Week1.day7.main import x_y_batches
from src.runtime import ROOT, fingerprint
from src.tokeniser.bpe import read_tokenizer


def load_data(settings):
    config = settings['data']
    directory = ROOT / config['processed_dir']
    metadata = json.loads((directory / 'metadata.json').read_text())
    tokenizer = ROOT / config['tokenizer_path']
    if metadata['tokenizer_sha256'] != fingerprint(tokenizer):
        raise ValueError('Dataset/tokenizer identity mismatch')
    vocab, _, _ = read_tokenizer(tokenizer)
    if metadata['vocab_size'] != len(vocab) or settings['model']['vocab_size'] != len(vocab):
        raise ValueError('Configured vocabulary does not match the prepared tokenizer')
    streams = []
    for split in ('train', 'val'):
        path = directory / f'{split}.bin'
        if fingerprint(path) != metadata['streams'][split]['sha256']:
            raise ValueError(f'{split} stream checksum mismatch')
        data = np.memmap(path, dtype=metadata['dtype'], mode='r')
        if len(data) != metadata['streams'][split]['tokens']:
            raise ValueError(f'{split} stream length mismatch')
        if len(data) <= settings['model']['context_length']:
            raise ValueError(f'{split} stream is too short')
        for start in range(0, len(data), 1024 * 1024):
            if data[start:start + 1024 * 1024].max() >= len(vocab):
                raise ValueError('Invalid token ID')
        streams.append(data)
    return *streams, metadata


def get_batch(data, batch_size, context_length, device):
    x, y = x_y_batches(data, batch_size, context_length)
    return x.to(device), y.to(device)
