from data_generator import ImageGenerator
from model import DeepViSe
from word2vec import load_word2vec_from_file

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
    word2vec = load_word2vec_from_file(args.word_embeddings)

    # Load dataset
    train_fnames = [str(p) for p in (dataset_dir_path/'train').glob('*/images/*.JPEG')]
