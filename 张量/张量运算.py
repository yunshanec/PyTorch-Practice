# -*- coding: utf-8 -*-
# @Time : 2021/06/25 15:14
# @Author : yunshan
# @File : 张量运算.py
import torch

# 1.张量的索引和切片
tensor = torch.ones(4,4)
tensor[:,1] = 0 #将第一列的数据全部赋值为0
print(tensor)

# 以下操作使用GPU运算
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(tensor)


# 2.张量的拼接(相同维度)
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

# 3.张量的乘积和矩阵乘法
# 张量的乘积和矩阵乘法
# 逐个元素相乘
multi_tensor = tensor.mul(tensor)
# 等价写法
multi_tensor = tensor * tensor
print("multi_tensor_result \n{}".format(multi_tensor))

# 张量与张量的矩阵乘法
matmul_tensor = tensor.matmul(tensor.T)
# 等价写法
matmul_tensor = tensor @ tensor
print("Matrix multi tensor result:\n {}".format(matmul_tensor))

# 4.自动赋值运算
# 自动赋值运算通常在方法后有 _ 作为后缀，例如x.copy_(y), x.t_() 操作会改变x的取值
result = tensor.add_(5)
print(result)