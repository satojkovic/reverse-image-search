from data_generator import ImageGenerator
from model import DeepViSe

import argparse
from tqdm import tqdm
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir_path', required=True, help='Path to tiny-imagenet directory')
    args = parser.parse_args()

    dataset_dir_path = pathlib.Path(args.dataset_dir_path)

    # Load dataset
    train_fnames = [str(p) for p in (dataset_dir_path/'train').glob('*/images/*.JPEG')]
