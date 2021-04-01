import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
from datasets import PETS
from datasets.pets_dataset import PetsDataset
from datasets.transform.ToTensor import ToTensor
from datasets.transform.RandomCrop import RandomCrop
from datasets.transform.Rescale import Rescale
from torch.utils.data import DataLoader
from utils.image import imgshow
from utils.sample import show_samples
from model.train import train_model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pets = PETS(root='data', download=True)
    train_data, train_labels, test_data, test_labels = pets.split_data_set()

    print('data', len(pets.data))
    print('targets', len(pets.targets))
    print('train_data', len(train_data))
    print('train_labels', len(train_labels))
    print('test_data', len(test_data))
    print('test_labels', len(test_labels))

    dataset = {
        'train': {
            'data': train_data,
            'targets': train_labels
        },
        'val': {
            'data': test_data,
            'targets': test_labels
        }
    }

    data_transforms = {
        'train': transforms.Compose([Rescale(256), RandomCrop(200), ToTensor()]),
        'val': transforms.Compose([Rescale(256), RandomCrop(200), ToTensor()])
    }

    pets_datasets = {
        x: PetsDataset(root_dir='', data=dataset[x]['data'], labels=dataset[x]['targets'], transform=data_transforms[x]) for x in ['train', 'val']
    }

    # show_samples(pets_dataset, 5)

    dataloaders = {
        x: DataLoader(pets_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    print(datasets['train'].classes)

    for i_batch, sample in enumerate(dataloaders['train']):
        label = sample["label"]
        image = sample["image"]

        print(f'{i_batch}.) label={label}, shape={image.shape}')

        if i_batch == 3:
            break

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=25)