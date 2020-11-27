import torchvision


import datasets

if __name__ == '__main__':
    dataset = datasets.PETS(root='data', download=True)
    train_data, train_labels, test_data, test_labels = dataset.split_data_set()

    print('data', len(dataset.data))
    print('targets', len(dataset.target))
    print('train_data', len(train_data))
    print('train_labels', len(train_labels))
    print('test_data', len(test_data))
    print('test_labels', len(test_labels))