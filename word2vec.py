def load_word2vec_from_file(path):
    word2vec = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word2vec[values[0]] = list(map(float, values[1:]))
    return word2vec
