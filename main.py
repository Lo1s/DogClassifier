import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import transforms, models
from torch.optim import lr_scheduler
from datasets import PETS
from datasets.pets_dataset import PetsDataset
from datasets.transform.ToTensor import ToTensor
from datasets.transform.RandomCrop import RandomCrop
from datasets.transform.Rescale import Rescale
from torch.utils.data import DataLoader
from PIL import Image
from utils.visualize import visualize_model
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

    dataset_sizes = {phase: len(pets_datasets[phase]) for phase in ['train', 'val']}

    # show_samples(pets_dataset, 5)

    dataloaders = {
        x: DataLoader(pets_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    for i_batch, sample in enumerate(dataloaders['train']):
        label = sample["label"]
        image = sample["image"]

        print(f'{i_batch}.) label={label}, shape={image.shape}')

        if i_batch == 3:
            break

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    class_to_idx = pets_datasets['train'].class_to_idx
    num_classes = len(pets_datasets['train'].classes)
    print(f'Number of classes: {num_classes}')
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    trained_model_path = 'data/PETS/trained/state_dict_model.pt'

    if os.path.exists(trained_model_path):
        model_ft.load_state_dict(torch.load(trained_model_path))
        model_ft.eval()

        test_image_path = 'data/PETS/eval/images/IMG_1243.jpg'
        test_image = Image.open(test_image_path)

        with torch.no_grad():
            outputs = model_ft(transforms.ToTensor()(test_image).unsqueeze(0))
            _, preds = torch.max(outputs, 1)

            print('-' * 10 + ' Result ' + '-' * 10)
            result_dict = {list(class_to_idx.keys())[i]: output.item() for i, output in enumerate(outputs[0])}

            dict(sorted(result_dict.items(), key=lambda item: item[1]))
            for item in result_dict:
                print(f'{item}: {result_dict[item]}')

    else:
        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_conv = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                                 device, num_epochs=25)

        torch.save(model_conv.state_dict(), 'state_dict_model.pt')

        visualize_model(model_conv, dataloaders, list(pets_datasets['val'].classes), device, 6)

