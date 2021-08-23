# -*- coding: utf-8 -*-
# @Time : 2021/06/29 14:03
# @Author : yunshan
# @File : train.py
from loading_data import Loading_data
import datetime
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import pandas as pd
from net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))


# 1.加载数据
loading_data = Loading_data()
dl_train = loading_data.dl_train
dl_valid = loading_data.dl_valid

# 2.搭建神经网络
net = Net()
model = net
model.to(device)

model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_loss_func = nn.BCELoss()
model_metric_func = lambda y_pred, y_true: roc_auc_score(
    y_true.data.cpu().numpy(), y_pred.data.cpu().numpy()
)
model_metric_name = "auc"


def train_step(model, features, labels):
    features = features.to(device)
    labels = labels.to(device)

    model.train()
    model_optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model_loss_func(predictions, labels)
    metric = model_metric_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    model_optimizer.step()

    return loss.item(), metric.item()


def valid_step(model, features, labels):
    features = features.to(device)
    labels = labels.to(device)

    model.eval()
    predictions = model(features)
    loss = model_loss_func(predictions, labels)
    metric = model_metric_func(predictions, labels)
    return loss.item(), metric.item()


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):
    metric_name = model_metric_name
    dfhistory = pd.DataFrame(
        columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name]
    )
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("==========" * 8 + "%s" % nowtime)

    for epoch in range(1, epochs + 1):
        # 1.训练循环
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(dl_train, 1):
            loss, metric = train_step(model, features, labels)
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(
                    ("[step = %d] loss: %.3f" + metric_name + ": %.3f")
                    % (step, loss_sum / step, metric_sum / step)
                )
        # 2.验证循环
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3.记录日志
        info = (
            epoch,
            loss_sum / step,
            metric_sum / step,
            val_loss_sum / val_step,
            val_metric_sum / val_step,
        )
        dfhistory.loc[epoch - 1] = info

        # 4.打印epoch级别日志
        print(
            (
                "\nEpoch = %d, loss = %.3f,"
                + metric_name
                + " = %.3f,val_loss = %.3f,"
                + "val_"
                + metric_name
                + " =%.3f"
            )
            % info
        )
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=======" * 8 + "%s" % nowtime)

    print("Finished Training...")
    return dfhistory


dfhistory = train_model(model,20, dl_train, dl_valid, log_step_freq=50)
