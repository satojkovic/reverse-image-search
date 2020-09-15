from data_generator import ImageGenerator
from model import DeepViSe

import argparse
from tqdm import tqdm
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir_path', required=True, help='Path to tiny-imagenet directory')
    parser.add_argument('--word_embeddings', required=True, help='Path to pretrained word embeddings')
    args = parser.parse_args()

    dataset_dir_path = pathlib.Path(args.dataset_dir_path)

    # Load pretrained word embeddings
    word2vec = {}
    with open(args.word_embeddings, 'r') as f:
        for line in f:
            values = line.split()
            word2vec[values[0]] = list(map(float, values[1:]))

    # Load dataset
    train_fnames = [str(p) for p in (dataset_dir_path/'train').glob('*/images/*.JPEG')]
