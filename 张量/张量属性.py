# -*- coding: utf-8 -*-
# @Time : 2021/06/25 15:09
# @Author : yunshan
# @File : 张量属性.py

# 获取张量的维数、数据类型以及它们所存储的设备
import torch
tensor = torch.rand(3,4)
print("Shape of tensor: {}".format(tensor.shape))
print("Datatype of tensor: {}".format(tensor.dtype))
print("Device tensor is stored on: {}".format(tensor.device))
