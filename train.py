import torch
from dataloader import DeviceDataLoader, DeviseDataset, to_device, load_pickle
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import DeviseModel, fit


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
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
    val_dataset = DeviseDataset(
        val_images, val_image_word_vecs, transform=transform
    )
    test_dataset = DeviseDataset(
        test_images, test_image_word_vecs, transform=transform
    )

    device = get_default_device()
    batch_size = 64
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 3e-3

    train_dl = DataLoader(train_dataset, batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2,
                        shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2,
                         shuffle=False, num_workers=0, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    model = to_device(DeviseModel(), device)

    for param in list(model.parameters()):
        param.requires_grad = True
    for param in list(model.parameters())[:-8]:
        param.requires_grad = False

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func=opt_func)
    torch.save(model.state_dict(), 'devise_resnet18.pth')
