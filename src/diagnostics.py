import platform

import torch

from src.model.config import GPTConfig
from src.model.gpt import GPT, parameter_counts
from src.runtime import select_device, settings_from_cli


def diagnostics(settings):
    print('Python:', platform.python_version())
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('PyTorch CUDA:', torch.version.cuda)
    device = select_device(settings)
    print('Selected device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(device))
        print('BF16 supported:', torch.cuda.is_bf16_supported())
    with torch.device('meta'):
        model = GPT(GPTConfig(**settings['model']))
    print('Configured parameter counts:', parameter_counts(model))


if __name__ == '__main__':
    diagnostics(settings_from_cli())
