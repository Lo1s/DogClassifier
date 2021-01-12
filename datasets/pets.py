import os
import tarfile
import torch
import numpy as np
import imghdr
import re
from pathlib import Path
from torchvision.transforms import ToTensor
from utils.data import download_file
from utils.image import imgshow, image_to_tensor
from collections import Counter


class PETS:
    """'The Oxford-IIIT https://www.robots.ox.ac.uk/~vgg/data/pets/' Pet Dataset.

    Args:
        root (string): Root dir
        download (bool, optional)
    """

    resources = [
        ('data', 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'),
        ('labels', 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
    ]

    project_path = Path.cwd()
    data_dir = 'data'
    data_path = project_path / data_dir
    data_file = 'data.pt'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(
            self,
            root: str,
            download: bool,
            print_progress=True,
            train2test_split_ratio=0.8
    ):
        self.root = root
        self.print_progress = print_progress
        self.train2test_split_ratio = train2test_split_ratio

        if download:
            self.download()

        dataset = torch.load(os.path.join(self.processed_folder, self.data_file))
        self.data, self.target = dataset['data'], dataset['labels']

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for key, url in self.resources:
            filepath = download_file(url, path=Path(self.raw_folder), print_progress=self.print_progress)

            print('Processing: ', filepath)
            dest = os.path.join(self.processed_folder, key)
            self.extract_file(filepath, Path(dest))

        img_dir = os.path.join(self.processed_folder, 'data', 'images')
        dataset = {
            'data': [],
            'labels': []
        }
        for file in os.listdir(img_dir):
            filepath = os.path.join(img_dir, file)
            extension = os.path.splitext(filepath)[1]
            if extension == '.jpg':
                dataset['data'].append(image_to_tensor(filepath))
                dataset['labels'].append(re.findall(r'(.+)_\d+.jpg$', file)[0])

        with open(os.path.join(self.processed_folder, self.data_file), 'wb') as f:
            torch.save(dataset, f)

        print('Done!')

    def split_data_set(self):
        keys = list(Counter(self.target).keys())
        counts = list(Counter(self.target).values())
        train_data, train_labels, test_data, test_labels = [], [], [], []

        for i in range(len(keys)):
            count = counts[i]
            key = keys[i]
            idx = [index for index, label in enumerate(self.target) if label == key]
            if count is not len(idx):
                raise RuntimeError(f'Size of the list of indexes for given breed={key} does not match')

            split = int(count * self.train2test_split_ratio)
            train_data.extend(map(self.data.__getitem__, idx[:split]))  # alternative: self.data[i] for i in idx[:split]
            train_labels.extend(map(self.target.__getitem__, idx[:split]))

            test_data.extend(map(self.data.__getitem__, idx[split:]))
            test_labels.extend(map(self.target.__getitem__, idx[split:]))

        if ((len(train_data) + len(test_data)) != len(self.data)) \
                or ((len(train_labels) + len(test_labels)) != len(self.target)):
            raise RuntimeError('Train + Test sizes does not match the total count')

        return train_data, train_labels, test_data, test_labels

    def extract_file(self, filepath, dest):
        if os.path.exists(dest):
            if self.print_progress: print('File already extracted, skipping: ', filepath)
            return
        else:
            os.makedirs(dest)

        tar = tarfile.open(filepath, 'r:gz')
        tar.extractall(dest)
