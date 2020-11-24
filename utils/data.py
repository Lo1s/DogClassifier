import gzip
import os
import shutil
from pathlib import Path

import idx2numpy
import requests
import validators


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


def get_mnist_dataset(print_progress=True):
    datasets = URLs.MNIST_DATASET

    for dataset in datasets:
        filepath = download_file(dataset, print_progress=print_progress)

        train_set = 'train' in str(filepath)
        images = 'images' in str(filepath)
        set_type = 'train' if train_set else 'test'
        data_type = 'images' if images else 'labels'
        dest = URLs.DATA_PATH / set_type / data_type
        if not os.path.exists(dest):
            os.makedirs(dest)

        file_dest = extract_file(str(filepath), dest, print_progress=print_progress)
        data[set_type][data_type] = load_idx_file(str(file_dest))

    return data


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


def extract_file(filepath, dest, print_progress=False):
    file_dest = dest / os.path.splitext(os.path.basename(filepath))[0]
    if os.path.exists(file_dest):
        if print_progress: print('File already extracted, skipping: ', file_dest)
        return file_dest

    with gzip.open(filepath, 'r') as file_in:
        with open(file_dest, 'wb') as file_out:
            shutil.copyfileobj(file_in, file_out)
            return file_dest


def is_already_downloaded(filepath):
    return os.path.exists(filepath)


def load_idx_file(filepath):
    return idx2numpy.convert_from_file(filepath)
