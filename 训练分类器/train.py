# -*- coding: utf-8 -*-
# @Time : 2021/06/28 09:46
# @Author : yunshan
# @File : 定义卷积神经网络.py
import time
import torch
import torch.nn as nn
from net import Net
from load_standard_CIFAR10_data import Load_Standard_CIFAR10_data as Load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}.".format(device))

# 1.加载数据
load_data = Load_data()
trainloader = load_data.trainloader

# 2.加载网络
# using gpu device
net = Net().to(device)

# 3.定义损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4.训练网络
start_time = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs,labels]
        # inputs,labels = data[0].to(device), data[1].to(device)
        inputs, labels = data

        # using gpu device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

PATH = "./data/cifar_net.path"
torch.save(net.state_dict(), PATH)

end_time = time.time()
print("Finished Training")
print("Spend {:.2f} seconds".format(end_time - start_time))
