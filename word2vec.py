import numpy as np

def load_word2vec_from_file(path):
    word2vec = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word2vec[values[0]] = list(map(float, values[1:]))
    vec_size = len(word2vec[list(word2vec.keys())[0]])
    return word2vec, vec_size

def load_wordnet(path):
    id2word, word2id = {}, {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.rstrip().split()
            wordnet_id = tokens[0]
            word = tokens[1].lower()
            id2word[wordnet_id] = word
            word2id[word] = wordnet_id
    return id2word, word2id

def get_vec_by_word(word, word2vec, vec_size):
    if word in word2vec:
        return word2vec[word]

    # `word` is not exist in word2vec because `word` is composite words
    vec = np.zeros((vec_size), dtype=np.float32)
    words = word.split('_')
    if len(words) == 1:
        return np.random.random((vec_size))
    for w in words:
        vec = vec + get_vec_by_word(w, word2vec, vec_size)
    return vec / len(words)

def get_word2vec_from_fname(fname, id2word, word2vec, vec_size):
    fnames = fname.split('/')
    if 'train' in fnames:
        index = fnames.index('train') + 1
        id = fnames[index]
        word = id2word[id]
        vec = get_vec_by_word(word, word2vec, vec_size)
        return vec

def get_word2vec_from_annotation(dataset_dir_path, id2word, word2vec, vec_size):
    valid_fnames = []
    valid_labels = []
    with open(dataset_dir_path/'val/val_annotations.txt') as f:
        for line in f:
            fname, id = line.strip().split('\t')[:2]
            fname = (dataset_dir_path/'val/images')/fname
            valid_fnames.append(fname)
            word = id2word[id]
            vec = get_vec_by_word(word, word2vec, vec_size)
            valid_labels.append(vec)
    return valid_fnames, valid_labels