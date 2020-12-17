from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch


def main():
    x = torch.ones(5,3,requires_grad=True)
    out.backward(torch.tensor(1.))
    print(x.grad)



if __name__ == '__main__':
    main()