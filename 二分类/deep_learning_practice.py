# -*- coding: utf-8 -*-
# @Time : 2021/06/28 16:21
# @Author : yunshan
# @File : deep_learning_practice.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device(device='cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device is {}.".format(device))



# 生成数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)


'''


# 搭建神经网络
net = nn.Sequential(
    nn.Linear(2,5),
    nn.Sigmoid(),
    nn.Linear(5,5),
    nn.Sigmoid(),
    nn.Linear(5,2),
    nn.Softmax(dim=1)
)

print(net)

# 配置损失函数和优化器
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
loss_func = nn.CrossEntropyLoss()
plt.ion() # 实时打印的过程

# 模型训练
num_epoch= 10000
for epoch in range(num_epoch):
    out = net(x) # 输入x得到预测值
    loss = loss_func(out,y) #计算损失，预测值和真实值的对比
    optimizer.zero_grad() # 梯度全降为0
    loss.backward() # 方向传递过程
    optimizer.step() # 以学习效率0.5来优化梯度
    #
    # if epoch % 100 == 0:
    #     print("epoch: {}, loss: {}".format(epoch,loss.data.item()))
    # 结果可视化
    if epoch % 50 == 0:
        plt.cla()
        prediction = torch.max(out,1)[1]
        prediction_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c = prediction_y,s=100,lw = 0,cmap="RdYlGn")
        accuracy = sum(prediction_y == target_y) /200
        plt.text(1.5,-4,"Accuracy=%.2f" % accuracy,fontdict={"size":20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()

# 网络保存
# 保存网络的所有参数
torch.save(net,'./net.pkl')
# 保存优化选项默认字典,不保存网络结构
torch.save(net.state_dict(),'./net_parameter.pkl')
'''
net1 = torch.load('./net.pkl')
prediction1 = net1(x)
plt.figure(1,figsize=(10,3))
plt.subplot(121)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
plt.show()
