from data_generator import ImageGenerator
from model import DeepViSe
from word2vec import get_vec_by_word, get_word2vec_from_annotation, get_word2vec_from_fname, load_word2vec_from_file, load_wordnet

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
    pattern = 'images/*.JPEG'
    train_fnames = [str(p) for p in (dataset_dir_path/'train').glob('*/' + pattern)]
    print(len(train_fnames))
    train_labels = [get_word2vec_from_fname(str(p), syn2word, word2vec, vec_size) 
                    for p in tqdm((dataset_dir_path/'train').glob(pattern))]
    train_gen = ImageGenerator(dataset_dir_path, train_fnames, train_labels,
                               classes_size=300, batch_size=64)
    print('The number of batches per epoch(train):', len(train_gen))

    val_fnames, val_labels = get_word2vec_from_annotation(
        dataset_dir_path, syn2word, word2vec, vec_size
    )
    val_gen = ImageGenerator(dataset_dir_path, val_fnames, val_labels, classes_size=300, batch_size=64)
    print('The number of batches per epoch(val):', len(val_gen))
