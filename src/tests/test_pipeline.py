import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.acquire import clean_text, strip_gutenberg, wiki_articles
from src.data.dataset import load_data
from src.data.prepare import document_batches, prepare
from src.runtime import ROOT, select_device
from src.smoke import smoke
from src.tokeniser.bpe import read_tokenizer, train_tokenizer, encode_bpe, decode_bpe
from src.train import train


class PipelineTests(unittest.TestCase):
    def settings(self, directory):
        settings = json.loads((ROOT / 'configs/debug.json').read_text())
        settings['data']['processed_dir'] = str(directory)
        settings['data']['tokenizer_path'] = str(directory / 'tokenizer.json')
        settings['checkpoint_dir'] = str(directory / 'checkpoints')
        return settings

    def test_tokenizer_serialization(self):
        text = 'Hello, world! A story in English. café 🙂\n' * 20
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tokenizer.json'
            vocab, merges, eod = train_tokenizer(text, 288, path)
            ids = encode_bpe(text, merges)
            original_vocab = vocab.copy()
            original_merges = list(merges.items())
            del vocab, merges
            vocab, merges, loaded_eod = read_tokenizer(path)
            self.assertEqual(vocab, original_vocab)
            self.assertEqual(list(merges.items()), original_merges)
            self.assertEqual(eod, loaded_eod)
            self.assertEqual(ids, encode_bpe(text, merges))
            self.assertEqual(text, decode_bpe(ids, vocab))

    def test_source_cache_replays_without_network(self):
        from src.data.acquire import cached_documents
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads((ROOT / 'configs/debug.json').read_text())['data']
            config['cache_dir'] = tmp
            docs = [{'text': 'A cached source document.', 'origin': 'test'}]
            with patch('src.data.acquire.source_documents', return_value=iter(docs)) as remote:
                self.assertEqual(list(cached_documents('fineweb', config)), docs)
                self.assertEqual(remote.call_count, 1)
            with patch('src.data.acquire.source_documents', side_effect=AssertionError('Network used')):
                self.assertEqual(list(cached_documents('fineweb', config)), docs)
            config['hf']['fineweb']['revision'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                list(cached_documents('fineweb', config))

    def test_cpu_smoke(self):
        smoke()

    def test_gpu_fails_without_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                select_device({'runtime': {'device': 'cuda'}})

    def test_cleaning_and_articles(self):
        body = 'This is an English paragraph with punctuation, and a second sentence. ' * 3
        raw = 'license\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n' + body + '\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense'
        self.assertEqual(clean_text(strip_gutenberg(raw)), body.strip())
        with self.assertRaises(ValueError):
            strip_gutenberg('unrecognized wrapper')
        self.assertEqual(clean_text('bad\x00text'), '')
        articles = list(wiki_articles([{'text': text} for text in
                                      ['\n', ' = First Article = \n',body,' == Section == \n',body,' = Second Article = \n',body]]))
        self.assertEqual(len(articles), 2)
        self.assertIn('Section', articles[0]['text'])

    def test_real_pipeline_with_local_sources(self):
        # Exercise the real mixture branch without contacting any source.
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            settings['runtime']['device'] = 'cuda'
            settings['data']['target_tokens'] = 3000
            docs = [{'text': text, 'origin': 'synthetic-test-source'} for text in
                    (ROOT / 'data/fixtures/english.txt').read_text().split('\n---\n')]
            def documents(source, config):
                return iter(dict(doc, text=source + ': ' + doc['text']) for doc in docs)
            with patch('src.data.prepare.cached_documents', side_effect=documents):
                metadata = prepare(settings)
            train_data, val_data, _ = load_data(settings)
            self.assertEqual(len(train_data) + len(val_data), 3000)
            self.assertEqual(set(metadata['sources']), {'fineweb', 'wikitext', 'gutenberg'})
            manifest = json.loads((Path(tmp) / 'documents.json').read_text())
            train_hashes = {d['sha256'] for d in manifest if d['split'] == 'train'}
            val_hashes = {d['sha256'] for d in manifest if d['split'] == 'val'}
            self.assertFalse(train_hashes & val_hashes)

    def test_identity_and_raw_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            prepare(settings)
            with patch('src.data.prepare.prepare', side_effect=AssertionError('raw text used')):
                load_data(settings)
            path = Path(tmp) / 'tokenizer.json'
            artifact = json.loads(path.read_text())
            artifact['version'] = 99
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_data(settings)

    def test_legacy_tokenizer_ids(self):
        from week2.day5.main import load_tokenizer
        path = ROOT / 'week2/day7/tokenizer.json'
        old_vocab, old_merges = load_tokenizer(path)
        vocab, merges, eod = read_tokenizer(path)
        self.assertEqual(vocab, old_vocab)
        self.assertEqual(list(merges.items()), list(old_merges.items()))
        self.assertIsNone(eod)

    def test_deduplication_after_rejected_batch(self):
        import random
        config = json.loads((ROOT / 'configs/debug.json').read_text())['data']
        text = 'This English text is a document that is long enough to survive the cleaning stage. ' * 3
        docs = iter([{'text': ''}] * 20 + [{'text': text}, {'text': text.upper()}])
        batches = list(document_batches(docs, 'test', config, set(), random.Random(42)))
        self.assertEqual(sum(len(a)+len(b) for a,b in batches), 1)

    def test_checkpoint_rejects_schedule_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            prepare(settings)
            train(settings, stop_after=1)
            settings['training']['max_steps'] += 1
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                train(settings)


if __name__ == '__main__':
    unittest.main()
