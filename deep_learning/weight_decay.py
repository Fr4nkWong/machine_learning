import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt


def calc_loss(y_hat, y):
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


def set_figsize(figsize=(3.5, 2.5)):
    # use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def plot_loss(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    # set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    # dataset
    num_train, num_test = 20, 100
    num_attr = 200
    batch_size = 1
    w, b = torch.ones(num_attr, 1)*0.01, 0.05
    x_mat = torch.randn(num_train+num_test, num_attr)
    y_vec = torch.matmul(x_mat, w) + b
    y_vec += torch.tensor(np.random.normal(0, 0.01, size=y_vec.size()), dtype=torch.float) # noise
    train_mat, test_mat = x_mat[:num_train,:], x_mat[num_train:,:]
    train_y, test_y = y_vec[:num_train], y_vec[num_train:]
    dataset = torch.utils.data.TensorDataset(train_mat, train_y)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # model
    lr, wd = 0.003, 3 # wd=0, no weight decay
    net = nn.Linear(num_attr, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减
    # training
    num_epochs = 100
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            output = net(X)
            ls = calc_loss(output, y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            ls.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(calc_loss(net(train_mat), train_y).mean().item())
        test_ls.append(calc_loss(net(test_mat), test_y).mean().item())
    plot_loss(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())

        