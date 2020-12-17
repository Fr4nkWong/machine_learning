import torch
import numpy as np

import linear_net


# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


if __name__ == "__main__":
    # datasets
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # model
    net = linear_net.LinearNet(1)
    torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.linear.bias, val=0)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # training
    num_epochs = 5000
    for epoch in range(1, num_epochs + 1):
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        loss.backward()
        optimizer.step()
        # ('epoch %d, loss: %f' % (epoch, loss.item()))
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    print(outputs)
    print(loss)