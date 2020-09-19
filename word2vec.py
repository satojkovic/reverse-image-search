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