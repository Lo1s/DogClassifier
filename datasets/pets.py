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

    PROJECT_PATH = Path.cwd()
    DATA_DIR = 'data'
    DATA_PATH = PROJECT_PATH / DATA_DIR

    def __init__(
            self,
            root: str,
            download: bool,
            print_progress=True
    ):
        self.root = root
        self.print_progress = print_progress

        if download:
            self.download()

        img_dir = os.path.join(self.processed_folder, 'data', 'images')
        training_set = {
            'data': [],
            'labels': []
        }
        for file in os.listdir(img_dir):
            filepath = os.path.join(img_dir, file)
            extension = os.path.splitext(filepath)[1]
            if extension == '.jpg':
                training_set['data'].append(image_to_tensor(filepath))
                training_set['labels'].append(re.findall(r'(.+)_\d+.jpg$', file)[0])

        print(training_set)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return os.path.exists(self.root)

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for key, url in self.resources:
            filepath = download_file(url, path=Path(self.raw_folder), print_progress=self.print_progress)

            print('Processing: ', filepath)
            dest = os.path.join(self.processed_folder, key)
            self.extract_file(filepath, Path(dest))

    def extract_file(self, filepath, dest):
        if os.path.exists(dest):
            if self.print_progress: print('File already extracted, skipping: ', filepath)
            return
        else:
            os.makedirs(dest)

        tar = tarfile.open(filepath, 'r:gz')
        tar.extractall(dest)


