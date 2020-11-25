import torchvision


# Press the green button in the gutter to run the script.
import datasets

if __name__ == '__main__':
    # torchvision.datasets.MNIST(root='/data', download=True)
    dataset = datasets.PETS(root='data', download=True)
