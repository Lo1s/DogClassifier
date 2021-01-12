import torch
from torch.utils.data import Dataset


class PetsDataset(Dataset):
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
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        image = self.data[index]
        label = self.labels[index]
        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


