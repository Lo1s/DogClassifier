import os
from pathlib import Path

from utils.data import download_file


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
            download: bool
    ):
        self.root = root

        if download:
            self.download()

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
            download_file(url, path=Path(self.raw_folder), print_progress=True)



