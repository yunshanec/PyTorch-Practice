# -*- coding: utf-8 -*-
# @Time : 2021/06/29 11:15
# @Author : yunshan
# @File : 利用自动微分和优化器求最小值.py
import torch

x = torch.tensor(0.0,requires_grad=True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr=0.01)

def f(x):
    result = a*torch.pow(x,2) + b*x + c
    return result

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()

print('y = ',f(x).data,'\n','x = ',x.data)