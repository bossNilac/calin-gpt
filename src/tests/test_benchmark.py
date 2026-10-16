import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.benchmark import check_encoding, evaluate, save_results
from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.runtime import seed_all


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(42)
        self.model = GPT(GPTConfig(vocab_size=256, context_length=8,
                                   n_layers=1, n_heads=2, d_model=16))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.ids = list(range(23))

    def test_full_text_and_windows(self):
        actual = evaluate(self.model, self.ids, self.vocab, 8, 255)
        ids = [255] + self.ids
        total = 0.0
        scored, end = 0, 8
        with torch.no_grad():
            while scored < len(self.ids):
                start = max(0, end - 8)
                for target in range(scored, end):
                    logits = self.model(torch.tensor([ids[start:target + 1]]))[0, -1]
                    total += F.cross_entropy(logits[None], torch.tensor([ids[target + 1]])).item()
                scored = end
                end = min(end + 4, len(self.ids))
        self.assertEqual(actual['tokens'], 23)
        self.assertEqual(actual['bytes'], 23)
        self.assertAlmostEqual(actual['loss'], total / 23, places=6)
        self.assertAlmostEqual(actual['bpb'], total / (23 * math.log(2)), places=6)
        self.assertEqual(actual, evaluate(self.model, self.ids, self.vocab, 8, 255))
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_512_token_context(self):
        model = GPT(GPTConfig(vocab_size=256, context_length=512,
                              n_layers=1, n_heads=2, d_model=16))
        forward = model.forward
        lengths = []
        def record(x):
            lengths.append(x.shape[1])
            return forward(x)
        ids = [i % 256 for i in range(1201)]
        with patch.object(model, 'forward', side_effect=record):
            result = evaluate(model, ids, self.vocab, 512, 255)
        self.assertEqual(result['tokens'], 1201)
        self.assertEqual(result['bytes'], 1201)
        self.assertEqual(lengths, [512, 512, 512, 512])

    def test_reference_output_interface(self):
        original = self.model.forward
        expected = evaluate(self.model, self.ids, self.vocab, 8, 255)
        with patch.object(self.model, 'forward', side_effect=lambda x: SimpleNamespace(logits=original(x))):
            actual = evaluate(self.model, self.ids, self.vocab, 8, 255)
        self.assertEqual(actual, expected)

    def test_byte_denominator_and_short_text(self):
        vocab = self.vocab.copy()
        vocab[0], vocab[1] = b'caf', 'é'.encode()
        check_encoding('café', [0, 1], vocab)
        result = evaluate(self.model, [0, 1], vocab, 8, 255)
        self.assertEqual(result['tokens'], 2)
        self.assertEqual(result['bytes'], 5)
        self.assertAlmostEqual(result['bpb'], result['loss'] * 2 / (5 * math.log(2)))
        with self.assertRaisesRegex(ValueError, 'exact test bytes'):
            check_encoding('wrong text', [0, 1], vocab)
        with self.assertRaisesRegex(ValueError, 'empty'):
            evaluate(self.model, [], vocab, 8, 255)

    def test_report_preserves_training_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.md'
            path.write_text('# Training notes\n\n## Benchmark Results\nOld score\n\n## File Hashes\nKeep this\n')
            save_results(path, '## Benchmark Results\nNew comparison\n')
            contents = path.read_text()
            self.assertIn('# Training notes', contents)
            self.assertIn('New comparison', contents)
            self.assertIn('## File Hashes\nKeep this', contents)
            self.assertNotIn('Old score', contents)
            save_results(path, '## Benchmark Results\nRepeated comparison\n')
            self.assertEqual(path.read_text().count('## Benchmark Results'), 1)


if __name__ == '__main__':
    unittest.main()
