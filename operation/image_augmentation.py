# -*- coding: utf-8 -*-
# @Time : 2021/06/30 10:41
# @Author : yunshan
# @File : image_enhancement.py
import os
import PIL.Image as Image
from torchvision import transforms as transforms

# outfile_path = './samples'
# img = Image.open('45.png')
# img.save(os.path.join(outfile_path,'test.png'))

# 1.按比例缩放
def resize_img(img,shape:tuple,item:int):
    new_img = transforms.Resize(shape)(img)
    print(f'{img.size}--->{new_img.size}')
    new_img.save(os.path.join(outfile_path,'{}.png'.format(item)))
    return  new_img

# shape = (128,256)
# resize_img(img,shape)

# 2.随机位置裁剪
def random_crop(img,box_length:int):
    new_img_R = transforms.RandomCrop(box_length)(img)
    new_img_R.save(os.path.join(outfile_path,'2_1.png'))
    new_img_C = transforms.CenterCrop(box_length)(img)
    new_img_C.save(os.path.join(outfile_path,'2_2.png'))

# random_crop(img,100)

# 3.水平垂直翻转
def H_V_Flip(img,p,item:int):
    new_img_H = transforms.RandomHorizontalFlip(p)(img)
    new_img_H.save(os.path.join(outfile_path,'{}_H.png'.format(item)))
    new_img_V = transforms.RandomVerticalFlip(p)(img)
    new_img_V.save(os.path.join(outfile_path,'{}_V.png'.format(item)))

# H_V_Flip(img,1)

# 4.角度旋转
def angle_rotation(img,angle:int,item:int):
    rotation_img = transforms.RandomRotation(angle)(img)
    rotation_img.save(os.path.join(outfile_path,'{}_rotation.png'.format(item)))

# angle_rotation(img,360)

# 5.亮度，对比度，饱和度，色度
def color_jitter(img,brightness,contrast,saturation,hue,item:int):
    new_img_B = transforms.ColorJitter(brightness)(img)
    new_img_B.save(os.path.join(outfile_path, '{}_brightness.png'.format(item)))

    new_img_C = transforms.ColorJitter(contrast)(img)
    new_img_C.save(os.path.join(outfile_path, '{}_contrast.png'.format(item)))

    new_img_S = transforms.ColorJitter(saturation)(img)
    new_img_S.save(os.path.join(outfile_path, '{}_saturation.png'.format(item)))

    new_img_H = transforms.ColorJitter(hue)(img)
    new_img_H.save(os.path.join(outfile_path, '{}_hue.png'.format(item)))

# color_jitter(img,2,3,1.5,1.5,item=10)

def gray_scale(img,p:float,item:int):
    new_img = transforms.RandomGrayscale(p=p)(img)
    new_img.save(os.path.join(outfile_path,'{}_gray.png'.format(item)))

# gray_scale(img,1)

def padding(img,item:int):
    new_img = transforms.Pad((0,(img.size[0] - img.size[1])//2))(img)
    new_img.save(os.path.join(outfile_path,'{}_padding.png'.format(item)))

# padding(img)

if __name__ == "__main__":
    shape = (128,256)
    brightness,contrast,saturation,hue = 2, 3, 1.5, 1.5
    outfile_path = './samples/0'
    for item in range(1,len(os.listdir('../data/train/0'))+1):
        print('loading...')
        new_img = Image.open('../data/train/0/{}.png'.format(item))
        #new_img = resize_img(new_img,shape,item) # 1
        H_V_Flip(new_img,1,item) # 2
        angle_rotation(new_img,90,item) # 1
        gray_scale(new_img,1,item) # 1
        color_jitter(new_img,brightness,contrast,saturation,hue,item) #4
        padding(new_img,item) # 1

    print("OK!")