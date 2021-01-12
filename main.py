import torchvision
import matplotlib.pyplot as plt

import datasets
from datasets.pets_dataset import PetsDataset
from utils.image import imgshow
from utils.sample import show_samples

if __name__ == '__main__':
    dataset = datasets.PETS(root='data', download=True)
    train_data, train_labels, test_data, test_labels = dataset.split_data_set()

    print('data', len(dataset.data))
    print('targets', len(dataset.target))
    print('train_data', len(train_data))
    print('train_labels', len(train_labels))
    print('test_data', len(test_data))
    print('test_labels', len(test_labels))

    pets_dataset = PetsDataset(root_dir='', data=train_data, labels=train_labels)

    show_samples(pets_dataset, 5)

    for i in range(len(pets_dataset)):
        sample = pets_dataset[i]
        label = sample["label"]
        image = sample["image"]

        print(f'{i}.) label={label}, shape={image.shape}')



