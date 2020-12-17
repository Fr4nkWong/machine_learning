import os

import torch
import numpy as np
import matplotlib.pyplot as plt


class LinearNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        y = self.linear(x.view(x.shape[0], -1)) 
        return y