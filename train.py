import torch
from dataloader import DeviceDataLoader, DeviseDataset
import pickle
import torchvision.transforms as transforms


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print('Loaded: {} {}'.format(pickle_path, data.shape))
    return data


if __name__ == '__main__':
    device = get_default_device()

    train_images = load_pickle('train_images.pkl')
    train_image_word_vecs = load_pickle('train_image_word_vecs.pkl')
    val_images = load_pickle('val_images.pkl')
    val_image_word_vecs = load_pickle('val_image_word_vecs.pkl')
    test_images = load_pickle('test_images.pkl')
    test_image_word_vecs = load_pickle('test_image_word_vecs.pkl')

    transform = transforms.Compose([
        transforms.Resize(size=(200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    train_dataset = DeviseDataset(
        train_images, train_image_word_vecs, transform=transform)
    valid_dataset = DeviseDataset(
        val_images, val_image_word_vecs, transform=transform
    )
    test_dataset = DeviseDataset(
        test_images, test_image_word_vecs, transform=transform
    )
