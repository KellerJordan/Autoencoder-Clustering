import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as T
from torchvision import datasets as dset


MNIST_TRN_TRANSFORM = T.Compose([
    T.ToTensor()
])
MNIST_TST_TRANSFORM = T.Compose([
    T.ToTensor()
])
CIFAR10_TRN_TRANSFORM = T.Compose([
    T.RandomCrop(28),
    T.ToTensor()
])
CIFAR10_TST_TRANSFORM = T.Compose([
    T.CenterCrop(28),
    T.ToTensor()                 
])

CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')

def get_mnist_dataset(trn_size=60000, tst_size=10000):
    trainset = dset.MNIST(root='./data', train=True,
                          download=True, transform=MNIST_TRN_TRANSFORM)
    trainset.train_data = trainset.train_data[:trn_size]
    trainset.train_labels = trainset.train_labels[:trn_size]
    testset = dset.MNIST(root='./data', train=False,
                         download=True, transform=MNIST_TST_TRANSFORM)
    testset.test_data = testset.test_data[:tst_size]
    testset.test_labels = testset.test_labels[:tst_size]
    return trainset, testset

def get_cifar10_dataset(trn_size=60000, tst_size=10000):
    trainset = dset.CIFAR10(root='./data', train=True,
                          download=True, transform=CIFAR10_TRN_TRANSFORM)
    trainset.train_data = trainset.train_data[:trn_size]
    trainset.train_labels = trainset.train_labels[:trn_size]
    testset = dset.CIFAR10(root='./data', train=False,
                         download=True, transform=CIFAR10_TST_TRANSFORM)
    testset.test_data = testset.test_data[:tst_size]
    testset.test_labels = testset.test_labels[:tst_size]
    return trainset, testset

def get_data_loader(trainset, testset, batch_size=128):
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False)
    return trainloader, testloader
