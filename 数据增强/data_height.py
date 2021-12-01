# -*- coding: utf-8 -*-
# @Time : 2021/11/03 11:21
# @Author : yunshan
# @File : data_height.py

import os
import torchvision.transforms as transforms
import PIL.Image as Image

out_file = "./out_cat"
img = Image.open("./cat.jpg")

"""
new_img = transforms.Resize((100,200))(img)
print(f'{img.size} ---> {new_img.size}')
new_img.save(os.path.join(out_file,"随机缩放.jpg"))
"""

"""
new_img1 = transforms.RandomResizedCrop(224)(img)
new_img1.save(os.path.join(out_file,"随机位置裁剪.jpg"))
new_img2 = transforms.CenterCrop(224)(img)
new_img2.save(os.path.join(out_file,"中心位置裁剪.jpg"))
"""
# Image._show(new_img2)
"""
new_img = transforms.RandomHorizontalFlip(p=1)(img) #p 表示概率
new_img.save(os.path.join(out_file,"水平翻转.jpg"))
new_img = transforms.RandomVerticalFlip(p=1)(img) #p 表示概率
new_img.save(os.path.join(out_file,"垂直翻转.jpg"))
"""

"""
new_img = transforms.RandomRotation(45)(img)
new_img.save(os.path.join(out_file,"随机角度旋转.jpg"))
"""

"""
new_img1 = transforms.ColorJitter(brightness=1)(img)
new_img1.save(os.path.join(out_file,"亮度.jpg"))

new_img2 = transforms.ColorJitter(contrast=1)(img)
new_img2.save(os.path.join(out_file,"对比度.jpg"))

new_img3 = transforms.ColorJitter(saturation=0.5)(img)
new_img3.save(os.path.join(out_file,"饱和度.jpg"))

new_img4= transforms.ColorJitter(hue=0.25)(img)
new_img4.save(os.path.join(out_file,"色度.jpg"))

new_img4= transforms.RandomGrayscale(p=0.75)(img)
new_img4.save(os.path.join(out_file,"灰度.jpg"))
"""

"""
img_size = img.size
new_img = transforms.Pad((0, (img_size[0] - img_size[1]) // 2))(img)
new_img.save(os.path.join(out_file, "Padding_正方形.jpg"))
Image._show(new_img)
"""

transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.RandomGrayscale(p=0.75)
])

import matplotlib.pyplot as plt

new_image = transform(img)
new_image.save(os.path.join(out_file, "new_image111.jpg"))


plt.subplot(1,2,1)
plt.imshow(img)
plt.title("original image")

plt.subplot(1,2,2)
plt.imshow(new_image)
plt.title("transform image")


plt.show()
