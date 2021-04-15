import fasttext as ft
import argparse
import os
from pathlib import Path
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasttext_file_path', required=True,
                        help='Path to fasttext bin file')
    parser.add_argument('--object_categories_path', required=True,
                        help='Path to 256_ObjectCategories directory')
    args = parser.parse_args()

    FASTTEXT_FILEPATH = Path(args.fasttext_file_path)
    ft_vecs = ft.load_model(str(FASTTEXT_FILEPATH))
    ft_words = ft_vecs.get_words(include_freq=True)
    ft_word_dict = {k: v for k, v in zip(*ft_words)}
    ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])

    with open('caltech256_categories_to_fasttext.txt', 'w') as f:
        for d in sorted(os.listdir(args.object_categories_path)):
            name = d.split('.')[1]
            if name == 'DS_Store':
                continue
            if name in ft_words:
                f.writelines(','.join([name, name]))
                f.writelines('\n')
            else:
                names = name.split('-')
                new_name = ''.join(names[:-1]) if re.match(r'\d+',
                                                           names[-1]) else ''.join(names)
                if new_name in ft_words:
                    f.writelines(','.join([name, new_name]))
                    f.writelines('\n')
                else:
                    print('Not exist in fasttext:', name)
