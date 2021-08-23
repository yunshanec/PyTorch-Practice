# -*- coding: utf-8 -*-
# @Time : 2021/06/28 11:08
# @Author : yunshan
# @File : test_net.py
import torch
import torchvision
from net import Net
from Image_show import imshow
from load_standard_CIFAR10_data import Load_Standard_CIFAR10_data as Load_data

# 加载测试数据
load_data = Load_data()
transform = load_data.transform
testset = load_data.testset
testloader = load_data.testloader
classes = load_data.classes

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

PATH = "./data/cifar_net.path"
net = Net()
net.load_state_dict(torch.load((PATH)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))
