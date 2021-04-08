from torch.utils.data import Dataset, DataLoader


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        for d in self.data_loader:
            yield to_device(d, self.device)

    def __len__(self):
        reutrn len(self.data_loader)
