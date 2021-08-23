# -*- coding: utf-8 -*-
# @Time : 2021/06/28 10:55
# @Author : yunshan
# @File : 根据册数数据测试网络.py

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from load_standard_CIFAR10_data import Load_Standard_CIFAR10_data as Load_data

load_data = Load_data()
transform = load_data.transform
testset = load_data.testset
testloader = load_data.testloader
classes = load_data.classes


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))
