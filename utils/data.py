import gzip
import os
import shutil
from pathlib import Path

import idx2numpy
import requests
import validators
import tarfile


class URLs:
    PROJECT_PATH = Path.cwd()
    DATA_DIR = 'data'
    DATA_PATH = PROJECT_PATH / DATA_DIR
    MNIST_TRAIN_IMAGES = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    MNIST_TRAIN_LABELS = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    MNIST_TEST_IMAGES = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    MNIST_TEST_LABELS = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    MNIST_DATASET = [MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, MNIST_TEST_IMAGES, MNIST_TEST_LABELS]
    RESULTS_DIR = 'results'
    RESULTS_PATH = PROJECT_PATH / RESULTS_DIR


data = {
    'train': {
        'images': None,
        'labels': None
    },
    'test': {
        'images': None,
        'labels': None
    }
}


def download_file(url, path=URLs.DATA_PATH, print_progress=False):
    if not os.path.exists(path):
        path = URLs.DATA_PATH
    if not validators.url(url):
        print(f'URL={url} is not valid')
        return

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path / os.path.basename(url)

    if is_already_downloaded(filepath):
        if print_progress: print('File already downloaded, skipping:', filepath)
    else:
        if print_progress: print('Downloading: ', url)
        response = requests.get(url)
        with filepath.open('wb') as f:
            f.write(response.content)

    return filepath


def is_already_downloaded(filepath):
    return os.path.exists(filepath)


def load_idx_file(filepath):
    return idx2numpy.convert_from_file(filepath)
