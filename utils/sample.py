import matplotlib.pyplot as plt

from utils.image import imgshow


def show_samples(dataset, num_samples):
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        label = sample["label"]

        ax = plt.subplot(1, num_samples, i + 1)
        plt.tight_layout()
        ax.set_title(label)
        ax.axis('off')
        imgshow(image)

        if i == num_samples - 1:
            plt.show()
            break