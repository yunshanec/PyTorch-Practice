import datetime
import torch
from torch import nn
import pandas as pd
from net import Net
from loading_data import Loading_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# 1.加载数据
loading_data = Loading_data()
dl_train = loading_data.dl_train

# 2.搭建神经网络
net = Net()
model = net
# 模型移到GPU上
model.to(device)

model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_loss_func = nn.BCELoss()


def train_step(model, features, labels):
    features = features.to(device)
    labels = labels.to(device)
    model.train()
    model_optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model_loss_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    model_optimizer.step()

    return loss.item()


def train_model(model, epochs, dl_train, log_step_freq):
    dfhistory = pd.DataFrame(columns=["epoch", "loss", "val_loss"])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=====" * 3 + "%s" % nowtime)
    for epoch in range(1, epochs + 1):
        # 1.训练循环
        loss_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(dl_train, 1):
            loss = train_step(model, features, labels)
            loss_sum += loss
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f") % (step, loss_sum / step))
        # 3.记录日志
        info = (
            epoch,
            loss_sum / step,
        )
        # 4.打印epoch级别日志
        print(("\nEpoch = %d, loss = %.3f,") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=======" * 8 + "%s" % nowtime)
    print("Finished Training...")
    return dfhistory


dfhistory = train_model(model, 3000, dl_train, log_step_freq=100)
print(model.state_dict().keys())
# 保存网路模型
torch.save(model.state_dict(), "midel_parameter_train3.pkl")
