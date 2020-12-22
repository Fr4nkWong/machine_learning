import torch
import numpy as np


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature, is_logistic=False):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 1)
        self.is_logistic = is_logistic

    def forward(self, x):
        y = self.linear(x)
        if self.is_logistic:
            y = 1/(1+exp(-y))
        return y