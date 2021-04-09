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
    print('fastText words:', len(ft_words))
