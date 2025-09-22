from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int
    input_dim: int  # flattened size


def _mnist_transforms():
    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def _cifar_transforms():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def get_dataloaders(dataset: str, batch_size: int, num_workers: int = 4, seed: int = 0, val_size: int = 5000) -> DataBundle:
    g = torch.Generator().manual_seed(seed)

    if dataset.lower() == 'mnist':
        train_tf, test_tf = _mnist_transforms()
        train_full = datasets.MNIST(root='data', train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root='data', train=False, download=True, transform=test_tf)
        # val split
        val_size = min(val_size, len(train_full))
        train_size = len(train_full) - val_size
        train, val = random_split(train_full, [train_size, val_size], generator=g)
        num_classes = 10
        input_dim = 1 * 28 * 28

    elif dataset.lower() == 'cifar10':
        train_tf, test_tf = _cifar_transforms()
        train_full = datasets.CIFAR10(root='data', train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root='data', train=False, download=True, transform=test_tf)
        val_size = min(val_size, len(train_full))
        train_size = len(train_full) - val_size
        train, val = random_split(train_full, [train_size, val_size], generator=g)
        num_classes = 10
        input_dim = 3 * 32 * 32

    elif dataset.lower() == 'cifar100':
        train_tf, test_tf = _cifar_transforms()
        train_full = datasets.CIFAR100(root='data', train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root='data', train=False, download=True, transform=test_tf)
        val_size = min(val_size, len(train_full))
        train_size = len(train_full) - val_size
        train, val = random_split(train_full, [train_size, val_size], generator=g)
        num_classes = 100
        input_dim = 3 * 32 * 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return DataBundle(train_loader, val_loader, test_loader, num_classes, input_dim)
