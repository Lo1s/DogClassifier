import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def imgshow(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))


def image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path)
    image = ToTensor()(image)
    return image


def image_to_ndarray(path: str) -> np.ndarray:
    try:
        image = Image.open(path)
        image.verify()
        image = Image.open(path)
        image_array = np.asarray(image)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise CorruptedImageException(f'Expected 3 dim image. Got shape={image_array.shape}')
        return image_array
    except(IOError, SyntaxError, AttributeError):
        print("Test")
        raise CorruptedImageException


class CorruptedImageException(Exception):
    """ Raised when image is corrupted or in unrecognizable format """
    pass
