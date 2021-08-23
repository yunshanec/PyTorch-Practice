# -*- coding: utf-8 -*-
# @Time : 2021/06/29 10:37
# @Author : yunshan
# @File : 利用backwar方法求倒数.py

'''
Pytorch一般通过反向传播backwa方法实现梯度计算，
该方法求得的梯度将存在对应自变量张良的grad属性下。

backward方法通常在一个标量张量上调用
如果调用的张量非标量，则要传入一个和它形状相同的gradient参数张量
相当于用该gradient参数张量与调用张量作向量点乘，得到的标量结果再反向传播。
'''
import torch

# f(x) = a*x**2 + b*x +c

# 1.标量的反向传播
x = torch.tensor(0.0,requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x +c

y.backward()
dy_dx = x.grad
print(dy_dx)

# 2.非标量的反向传播
x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x +c

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])

print("x:\n",x)
print("y:\n",y)
y.backward(gradient = gradient)
x_grad = x.grad
print("x_grad:\n",x_grad)

# 3.非标量的反向传播可以用标量的反向传播实现
x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x +c

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
z = torch.sum(y*gradient)

print("x:",x)
print("y:",y)
z.backward()
x_grad = x.grad
print("x_grad:\n",x_grad)












