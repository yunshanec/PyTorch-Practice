# -*- coding: utf-8 -*-
# @Time : 2021/06/29 16:26
# @Author : yunshan
# @File : tets_result.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from net import Net


# 1.加载数据
transform_valid = transforms.Compose([transforms.ToTensor()])
ds_valid = datasets.ImageFolder(
    "../data/test/",
    transform=transform_valid,
    target_transform=lambda t: torch.tensor([t]).float(),
)
dl_valid = DataLoader(ds_valid, batch_size=16, shuffle=True, num_workers=3)


net_clone = Net()
net_clone.load_state_dict(torch.load('./midel_parameter_train3.pkl'))

def predict(model,dl):
    model.eval()
    result = torch.cat([model.forward(t[0]) for t in dl])
    return result.data
y_pred_probs = predict(net_clone,dl_valid)
y_pred_probs = torch.where(y_pred_probs>0.5,
                           torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs)
                           )
print(y_pred_probs)