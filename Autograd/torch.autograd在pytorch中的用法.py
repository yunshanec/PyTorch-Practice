# -*- coding: utf-8 -*-
# @Time : 2021/06/25 16:21
# @Author : yunshan
# @File : torch.autograd在pytorch中的用法.py

import torch, torchvision

# 从torchvision中加载预训练的resnet18模型
model = torchvision.models.resnet18(pretrained=True)
# 创建一个随机数据张量用来表示具有3个通道的单个图像,高度和宽度为64
data = torch.rand(1, 3, 64, 64)
# 图像对应的label初始化为一些随机值
labels = torch.rand(1, 1000)

# 通过模型的每一层运行输入数据以进行预测
prediction = model(data)  # forward pass
# 使用模型的预测和相应的标签来计算误差（loss）
loss = (prediction - labels).sum()
# 通过网络反向传播此误差
loss.backward()  # backward pass

# Autograd会为每个模型参数计算梯度并将其存储在参数.grad属性中

# 加载一个优化器，在本例子中为SGD，学习率0.01,动量为0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# 调用.step()启动梯度下降。优化器通过.grad中存储的梯度来调整每个参数
optim.step()  # gradient descent
