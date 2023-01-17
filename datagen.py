import zipfile
from pathlib import Path

import requests

import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize

from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.imagenet import ImageNet

from tqdm import tqdm


def load_mnist(
    batch_size: int = 64, shuffle: bool = True, root: str = "data"
) -> tuple[MNIST, MNIST]:
    """Load MNIST Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        root (str, optional): Path to store the data. Defaults to "data".

    Returns:
        tuple[MNIST, MNIST]: (train_data, test_data)
    """

    try:
        train = MNIST(
            root=root,
            train=True,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # ),
        )

        test = MNIST(
            root=root,
            train=False,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )
    except RuntimeError:
        train = MNIST(
            root=root,
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # ),
        )

        test = MNIST(
            root=root,
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data

def load_cifar10(
    batch_size: int = 128, shuffle: bool = True, root: str = "data"
) -> tuple[CIFAR10, CIFAR10]:
    """Load CIFAR10 Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        root (str, optional): Path to store the data. Defaults to "data".

    Returns:
        tuple[CIFAR10, CIFAR10]: (train_data, test_data)
    """

    try:
        train = CIFAR10(
            root=root,
            train=True,
            download=False,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # ),
        )

        test = CIFAR10(
            root=root,
            train=False,
            download=False,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        )
    except RuntimeError:
        train = CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # ),
        )

        test = CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data
