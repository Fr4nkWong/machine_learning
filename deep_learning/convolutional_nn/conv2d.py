import numpy as np
import torch
import torch.nn as nn


def corr2d(x_mat, conv_kernel):
    """
    2-dimension cross-correlation
    """
    h, w = x_mat.shape
    k_h, k_w = conv_kernel.shape
    y_mat = torch.zeros((h-k_h+1, w-k_w+1))
    for i in range(y_mat.shape[0]):
        for j in range(y_mat.shape[1]):
            y_mat[i][j] = (x_mat[i:i+k_h, j:j+k_w]*conv_kernel).sum()
    return y_mat


class Conv2d(nn.Module):
    """
    2-dimension convolutional layer
    """
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.parameter.Parameter(torch.randn(kernel_size))
        self.bias = nn.parameter.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == "__main__":
    # dataset
    x_mat = torch.ones(6,8)
    x_mat[:, 2:6] = 0
    conv_kernel = torch.tensor([[1,-1]])
    y_mat = corr2d(x_mat, conv_kernel)
    # training
    conv2d = Conv2d(kernel_size=(1, 2))
    step = 20
    lr = 0.01
    for i in range(step):
        y_hat = conv2d(x_mat)
        l = ((y_hat - y_mat) ** 2).sum()
        l.backward()
        conv2d.weight.data -= lr * conv2d.weight.grad
        conv2d.bias.data -= lr * conv2d.bias.grad
        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)
        if (i + 1) % 5 == 0:
            print('Step %d, loss %.3f' % (i + 1, l.item()))
    print("weight: ", conv2d.weight.data)
    print("bias: ", conv2d.bias.data)
