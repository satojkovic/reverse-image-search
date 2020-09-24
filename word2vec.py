import numpy as np

def load_word2vec_from_file(path):
    word2vec = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word2vec[values[0]] = list(map(float, values[1:]))
    return word2vec

def load_wordnet(path):
    syn2word, word2syn = {}, {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.rstrip().split()
            synset = tokens[0]
            word = tokens[1].lower()
            syn2word[synset] = word
            word2syn[word] = synset
    return syn2word, word2syn

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
