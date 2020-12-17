import torch
import numpy as np


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y