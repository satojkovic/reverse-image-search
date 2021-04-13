import fasttext as ft
import argparse
from pathlib import Path
import numpy as np
import pickle


def dump(pickle_path, data):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print('Dumped:', pickle_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasttext_path', required=True,
                        help='Path to fasttext dir')
    parser.add_argument('--object_categories_path', required=True,
                        help='Path to 256_ObjectCategories directory')
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

    test_classes = ["bear", "piano", "laptop", "syringe",
                    "tomato", "calculator", "rifle", "dog", "floppy", "octopus"]
    CALTECH256_PATH = Path(args.object_categories_path)
    classes = []
    train_images = []
    train_image_word_vecs = []
    test_images = []
    test_image_word_vecs = []
    for dir_name in CALTECH256_PATH.iterdir():
        class_name = dir_name.name.rpartition('.')[-1]
        if not class_name in caltech256_to_fasttext:
            continue
        class_name = caltech256_to_fasttext[class_name]
        class_images = list(dir_name.iterdir())
        if class_name in test_classes:
            test_images += class_images
            test_image_word_vecs += [lc_vec_d[class_name]] * len(class_images)
        else:
            classes.append(class_name)
            train_images += class_images
            train_image_word_vecs += [lc_vec_d[class_name]] * len(class_images)
    train_image_word_vecs = np.stack(train_image_word_vecs)
    test_image_word_vecs = np.stack(test_image_word_vecs)
    print('train_image_word_vecs.shape {}, test_image_word_vecs.shape {}'.format(
        train_image_word_vecs.shape, test_image_word_vecs.shape))

    # Train : Val = 0.75 : 0.25
    all_ids = np.arange(len(train_images))
    np.random.shuffle(all_ids)
    num_train_ids = int(len(all_ids) * 0.75)
    train_ids, val_ids = all_ids[:num_train_ids], all_ids[num_train_ids:]
    dump('train_images.pkl', np.array(train_images)[train_ids])
    dump('val_images.pkl', np.array(train_images)[val_ids])
    dump('test_images.pkl', np.array(test_images))
    dump('train_image_word_vecs.pkl', train_image_word_vecs[train_ids])
    dump('val_image_word_vecs.pkl', train_image_word_vecs[val_ids])
    dump('test_image_word_vecs.pkl', test_image_word_vecs)
