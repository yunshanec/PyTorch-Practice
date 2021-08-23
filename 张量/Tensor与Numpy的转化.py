# -*- coding: utf-8 -*-
# @Time : 2021/06/25 15:45
# @Author : yunshan
# @File : Tensor与Numpy的转化.py
import numpy as np
import torch

if torch.cuda.is_available():
    pass

'''

# 张量和Numpy array 数组在cpu上可以共用一块内存区域
# 改变其中一个另一个也会随之改变

# 1.由tensor张量变换为Numpy array 数组

t = torch.ones([5,5])
print("tensor: {}".format(t))
n = t.numpy()
print("numpy:{}".format(n))

# 修改张量的值，则Numpy array数组值也会随之改变
t.add_(4)
print(t)
print(n)

'''
# 2.由Numpy array数组转换为tensor张量
n = np.ones(5)
t = torch.from_numpy(n)
print(n)
print(t)

# 修改Numpy array 数组的值，则张量值也会随之改变
result = np.add(n, 1, out=n)
print(n)
print(t)
