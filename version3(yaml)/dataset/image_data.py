import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class cifar_dataset(Dataset):
    def __init__(self, train=True):
        super(cifar_dataset, self).__init__()
        data = CIFAR10(
            root="../CIFAR10/",
            train=train,
            download=True
        )

        self.images = [np.array(d[0]) for d in data]
        self.labels = [d[1] for d in data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "images": torch.tensor(self.images[idx]).transpose(0, 2)/255,
            "labels": self.labels[idx]
        }
