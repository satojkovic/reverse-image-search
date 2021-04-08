import torch
from dataloader import DeviceDataLoader


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    device = get_default_device()
