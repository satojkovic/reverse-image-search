from data_generator import ImageGenerator
from model import DeepViSe
from word2vec import get_word2vec_from_fname, load_word2vec_from_file, load_wordnet

import argparse
from tqdm import tqdm
import pathlib
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir_path', required=True, help='Path to tiny-imagenet directory')
    parser.add_argument('--word_embeddings', required=True, help='Path to pretrained word embeddings')
    args = parser.parse_args()

    dataset_dir_path = pathlib.Path(args.dataset_dir_path)

    # Load pretrained word embeddings
    word2vec, vec_size = load_word2vec_from_file(args.word_embeddings)
    if vec_size == 0:
        print('Embedding size:', vec_size)
        sys.exit(-1)

    # Load wordnet
    syn2word, word2syn = load_wordnet(dataset_dir_path/'words.txt')

    # Load dataset
    pattern = '*/images/*.JPEG'
    train_fnames = [str(p) for p in (dataset_dir_path/'train').glob('*/images/*.JPEG')]
    train_labels = [get_word2vec_from_fname(str(p), syn2word, word2vec, vec_size) 
                    for p in tqdm((dataset_dir_path/'train').glob(pattern))]
    print(train_fnames[0], train_labels[0])