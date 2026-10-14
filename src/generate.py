import torch

from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.runtime import ROOT, fingerprint, seed_all, select_device, settings_from_cli
from src.tokeniser.bpe import decode_generated, encode_bpe, read_tokenizer


@torch.no_grad()
def generate(settings):
    device = select_device(settings)
    seed_all(settings['data']['seed'])
    checkpoint = torch.load(ROOT / settings['checkpoint_dir'] / 'checkpoint.pt',
                            map_location=device, weights_only=True)
    path = ROOT / settings['data']['tokenizer_path']
    identity = checkpoint.get('identity')
    if identity is None or identity['tokenizer'] != fingerprint(path):
        raise ValueError('Checkpoint/tokenizer identity mismatch')
    vocab, merges, eod = read_tokenizer(path)
    config = GPTConfig(**checkpoint['model_config'])
    if config.vocab_size != len(vocab):
        raise ValueError('Checkpoint vocabulary mismatch')
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    options = settings['generation']
    ids = encode_bpe(options['prompt'], merges)
    if not ids:
        raise ValueError('Generation prompt must not be empty')
    for _ in range(options['max_new_tokens']):
        x = torch.tensor([ids[-config.context_length:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        temperature = options['temperature']
        token = (logits.argmax() if temperature == 0 else
                 torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)).item()
        if token == eod:
            break
        ids.append(token)
    return decode_generated(ids, vocab, eod)


if __name__ == '__main__':
    print(generate(settings_from_cli()))
