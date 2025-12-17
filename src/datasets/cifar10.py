
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_transforms(is_training=True):
    """
    Standard CIFAR-10 transforms.
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])

def load_cifar10(data_root, batch_size=128, num_workers=2, download=True):
    """
    Returns train and test dataloaders for CIFAR-10.
    """
    train_transform = get_cifar10_transforms(is_training=True)
    test_transform = get_cifar10_transforms(is_training=False)
    
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=download, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=download, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    return trainloader, testloader, classes
