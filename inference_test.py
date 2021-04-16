from model import DeviseModel, predict_dl_batch
from torchsummary import summary
import torch
from annoy import AnnoyIndex
from dataloader import load_pickle, DeviseDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import fasttext as ft
import matplotlib.pyplot as plt


def show_preds(idxs, dataset, images):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasttext_file_path', required=True,
                        help='Path to fasttext bin file')
    parser.add_argument('--query', required=True, help='Qeury word')
    args = parser.parse_args()

    model = DeviseModel()
    model.load_state_dict(torch.load('devise_resnet18.pth'))
    model.eval()

    batch_size = 64
    val_images = load_pickle('val_images.pkl')
    val_image_word_vecs = load_pickle('val_image_word_vecs.pkl')
    transform = transforms.Compose([
        transforms.Resize(size=(200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    val_dataset = DeviseDataset(
        val_images, val_image_word_vecs, transform=transform
    )
    val_dl = DataLoader(val_dataset, batch_size*2,
                        shuffle=False, num_workers=0, pin_memory=True)
    pred_word_vecs = predict_dl_batch(val_dl, model)

    # ANN oh yeah
    ann = AnnoyIndex(model.embed_dim, metric='euclidean')
    ntree = 100
    for i, pred_word_vec in enumerate(pred_word_vecs):
        ann.add_item(i, pred_word_vec)
    _ = ann.build(ntree)

    ft_vecs = ft.load_model(args.fasttext_file_path)
    vec = ft_vecs.get_word_vector(args.query)
    idxs = ann.get_nns_by_vector(vec, 9)
    show_preds(idxs, val_dataset, val_images)
