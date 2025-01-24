import torch


def select_device(device):
    if device is not None:
        return device

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device
