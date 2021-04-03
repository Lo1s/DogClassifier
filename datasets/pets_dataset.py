from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class PetsDataset(VisionDataset):
    """ PETS dataset """

    def __init__(self, root_dir, data, labels, transform=False):
        """
        Args:
            root_dir: Directory with all the images
            transform (callable, Optional): Optional transform to be applied on a sample
            data: train/test data
        """
        self.root_dir = root_dir
        self.data = data
        self.classes, self.class_to_idx = self._find_classes(labels)
        self.labels = [self.class_to_idx[label] for label in labels]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _find_classes(self, labels) -> Optional[Tuple[set, dict]]:
        if labels is None:
            return

        classes = set(labels)
        return classes, {class_name: i for i, class_name in enumerate(classes)}

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        image = self.data[index]
        label = self.labels[index]
        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


