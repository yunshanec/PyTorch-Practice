# -*- coding: utf-8 -*-
# @Time : 2021/06/29 13:26
# @Author : yunshan
# @File : loading_data.py
from matplotlib import pyplot as plt
import torch

from torch.utils.data import DataLoader
from torchvision import transforms,datasets

class Loading_data:
    def __init__(self):
        self.transform_train = transforms.Compose([transforms.ToTensor()])
        self.transform_valid = transforms.Compose([transforms.ToTensor()])

        self.ds_train = datasets.ImageFolder('../data/train/',
                                        transform=self.transform_train,target_transform=lambda t:torch.tensor([t]).float()
                                        )
        self.ds_valid = datasets.ImageFolder('../data/train/',
                                        transform=self.transform_train,target_transform=lambda t:torch.tensor([t]).float()
                                        )

        self.dl_train = DataLoader(self.ds_train,batch_size=8,shuffle=True,num_workers=5)
        self.dl_valid = DataLoader(self.ds_valid,batch_size=8,shuffle=True,num_workers=5)

    def imgshow(self):
        plt.figure(figsize=(8,8))
        for i in range(9):
            img,label = self.ds_train[i]
            img = img.permute(1,2,0)
            ax = plt.subplot(3,3,i+1)
            ax.imshow(img.numpy())
            ax.set_title("label = {}".format(label.item()))
            # ax.set_xticks([])
            # ax.set_yticks([])
        plt.show()

if __name__ == '__main__':
    loading_data = Loading_data()
    print(type(loading_data.ds_train))

    for x,y in loading_data.dl_train:
        print(x.shape,y.shape)
        break
    loading_data.imgshow()
