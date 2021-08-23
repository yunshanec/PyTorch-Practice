# -*- coding: utf-8 -*-
# @Time : 2021/06/30 09:11
# @Author : yunshan
# @File : img2torch.py

# opencv 读取图片后的维度是B,H,W,C并且读取的是BGR

import cv2
import numpy as np
import torch

class Img2Torch:
    def __init__(self):
        super(Img2Torch, self).__init__()

    # 读取RGB图像
    def rgb_img(self,img_dir):
        img = cv2.imread(img_dir)
        rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return rgb_img

    # resize RGB图像大小
    def resize_rgb_img(self,rgb_img,new_size):
        resized_img = cv2.resize(rgb_img,new_size,interpolation=cv2.INTER_LINEAR)
        return resized_img

    # 图像转换成张量
    def img2torch(self,RGB_IMG):
        tensor_out = torch.from_numpy(np.transpose(RGB_IMG,(2,0,1))).to('cuda').div(255.0).unsqueeze(0)
        return tensor_out

    # 张量转换成图像
    def torch2img(self,Tensor):
        img_out = np.transpose(Tensor.numpy(), (1, 2, 0))
        return img_out


if __name__ == '__main__':
    img_dir = './data/train/0/1.png'
    new_size = (128,256)

    image_to_torch = Img2Torch()
    new_img = image_to_torch.rgb_img(img_dir)
    resize_new_img = image_to_torch.resize_rgb_img(new_img,new_size)

    tensor_out = image_to_torch.img2torch(new_img)
    tensor_out1 = image_to_torch.img2torch(resize_new_img)

    print(tensor_out)
    print(tensor_out1)



