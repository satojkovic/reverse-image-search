from torch.utils.data import Dataset
from PIL import Image
import pickle


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
        return len(self.data_loader)


class DeviseDataset(Dataset):
    def __init__(self, images, vecs, transform=None):
        self.images = images
        self.vecs = vecs
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_vec = self.vecs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_vec


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print('Loaded: {} {}'.format(pickle_path, data.shape))
    return data
