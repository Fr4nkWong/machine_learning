import sys

import numpy as np
import torch
from torch import nn

sys.path.append("..") 
import d2lzh_pytorch as d2l


def load_data_fashion_mnist(batch_size, root='~/Downloads/Datasets/FashionMNIST'):
    transform = torchvision.transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    # load data quickly
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


if __name__ == '__main__':
    # dataset
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # model
    net = nn.Sequential(
        d2l.FlattenLayer(), # change x shape
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
    )
    for params in net.parameters():
        nn.init.normal_(params, mean=0, std=0.01)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # training
    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
    n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
    features = torch.randn((n_train + n_test, 1))
    poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
    labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
    print(true_w[0] * poly_features[:, 0])