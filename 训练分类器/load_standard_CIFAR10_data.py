# -*- coding: utf-8 -*-
# @Time : 2021/06/28 11:33
# @Author : yunshan
# @File : load_standard_CIFAR10_data.py

import torch
import torchvision
import torchvision.transforms as transforms

# 1.加载并标准化CIFAR10数据
class Load_Standard_CIFAR10_data:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=self.transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=False, transform=self.transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
