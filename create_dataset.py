import fasttext as ft
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasttext_path', required=True,
                        help='Path to fasttext dir')
    args = parser.parse_args()

    FASTTEXT_PATH = Path(args.fasttext_path)
    ft_vecs = ft.load_model(str(FASTTEXT_PATH / 'cc.en.300.bin'))
    ft_words = ft_vecs.get_words(include_freq=True)
    ft_word_dict = {k: v for k, v in zip(*ft_words)}
    ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])
    lc_vec_d = {w.lower(): ft_vecs.get_word_vector(w) for w in ft_words}
    print('fastText words:', len(ft_words))

    # Load mapping file
    caltech256_to_fasttext = {}
    with open('caltech256_categories_to_fasttext.txt', 'r') as f:
        for line in f:
            elems = line.strip().split(',')
            caltech256_to_fasttext[elems[0]] = elems[1]
    print('Num. of valid categories:', len(caltech256_to_fasttext))
