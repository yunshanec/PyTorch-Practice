# -*- coding: utf-8 -*-
# @Time : 2021/06/25 14:08
# @Author : yunshan
# @File : 张量.py
import torch

# 构造一个5*3矩阵，不初始化
# x = torch.empty(5,3)
# print(x)

# 构造一个随机初始化的矩阵
# x = torch.rand(5,3)
# print(x)

# 构造一个矩阵全为0，而且数据类型是long的矩阵
# x = torch.zeros(5,3,dtype=torch.long)
# print(x)

# 直接生成张量
data = [[1,2],[3,4]]
x_data= torch.tensor(data)
# print(x_data)

# 通过已有的张量来生成新的张量
# 新的张量将继承已有张量的数据属性（结构、类型），也可以重新指定新的数据类型
# x_ones= torch.ones_like(x_data) #保留x_data的属性
# print(f'Ones Tensor: \n {x_ones} \n')
# x_rand = torch.rand_like(x_data,dtype=torch.float) # 重写x_data的数据类型int--->float
# print(f'Random Tensor: \n {x_rand} \n')


# 通过指定数据维度来生成张量
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensors = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensors)


