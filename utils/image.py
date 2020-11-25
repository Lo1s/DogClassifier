import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def imgshow(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))
    plt.show()


def image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path)
    image = ToTensor()(image)
    return image
