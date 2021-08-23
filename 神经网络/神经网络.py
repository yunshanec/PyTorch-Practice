# -*- coding: utf-8 -*-
# @Time : 2021/06/25 17:03
# @Author : yunshan
# @File : 神经网络.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel ,6 output channels, 3*3 square convolution
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3)
        # an affine仿射operation: y = wx + b
        self.fc1 = nn.Linear(16*6*6,120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # If the size is square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    net = Net()
    # # 模型的可学习参数
    # model_learn_params =list(net.parameters())
    # print(len(model_learn_params))
    # print(model_learn_params[0].size()) # conv1's weight

    input = torch.randn(1,1,32,32)
    print(input)

    # # 使用随机梯度将所有参数和反向传播的梯度缓冲区归零
    # net.zero_grad()
    # output.backwarf(torch.randn(1,10))

    # 损失函数
    # 损失函数采用一对（输出，目标）输入，并计算一个值，该值估计输出与目标之间的距离
    output = net(input)
    print(output)
    target = torch.randn(10)
    target = target.view(1,-1)
    # cc.MSELoss()计算输入和目标之间的均方误差
    criterion = nn.MSELoss()
    loss = criterion(output,target)

    print(loss.grad_fn) #MSELoss
    print(loss.grad_fn.next_functions[0][0]) # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU

    # 反向传播
    # 要反向传播误差，我们要做的只是对loss.backward().不过，需要先清除现有的
    # 梯度，否则梯度将累积到现有的梯度中

    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # 更新权重
    # 使用随机梯度下降（SGD）规则
















