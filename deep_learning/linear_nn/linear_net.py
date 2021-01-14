import torch
import torch.nn as nn
import numpy as np


class LinearNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        y = self.linear(x.view(x.shape[0], -1)) 
        return y


class LogisticLayer(nn.Module):
    def __init__(self, **kwargs):
        super(LogisticLayer, self).__init__(**kwargs)

    def forward(self, x):
        y = 1/(1+torch.exp(-x))
        return y
