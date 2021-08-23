# -*- coding: utf-8 -*-
# @Time : 2021/06/25 16:48
# @Author : yunshan
# @File : Autograd的微分.py
import torch

x = torch.tensor(0.0,requires_grad=True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x +c

# create_graph = True 将允许创建更高阶的导数
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx,x)[0]
print(dy2_dx2.data)



x1 = torch.tensor(1.0,requires_grad=True)
x2 = torch.tensor(2.0,requires_grad=True)

y1 = x1*x2
y2 = x1+x2

# 允许同时对多个自变量求导数
(dy1_dx1,dy1_dx2) = torch.autograd.grad(outputs=y1,inputs=[x1,x2],retain_graph=True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1,dy12_dx2) = torch.autograd.grad(outputs=[y1,y2],inputs=[x1,x2])
print(dy12_dx1,dy12_dx2)