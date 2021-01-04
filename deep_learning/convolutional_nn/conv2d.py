import numpy
import torch
import torch.nn


def corr2d(input_mat, conv_kernel):
    """
    cross-correlation
    """
    heigh1, width1 = input_mat.shape
    heigh2, width2 = conv_kernel.shape
    output_mat = torch.zeros((heigh1-heigh2+1, width1-width2+1))
    for i in range(output_mat.shape[0]):
        for j in range(output_mat.shape[1]):
            output_mat[i][j] = (input_mat[i:i+heigh2, j+width2]*conv_kernel).sum()
    return output_mat


class Conv2d(nn.Module):
    """
    2-dimension convolutional layer
    """
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameters(torch.randn(kernel_size))
        self.bias = nn.Parameters(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == "__main__":
    pass