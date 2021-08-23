# -*- coding: utf-8 -*-
# @Time : 2021/07/08 17:19
# @Author : yunshan
# @File : load_progress_data.py
import pandas as pd
landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n,1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)

print(f'Image name:{img_name}')
print(f'Landmarks shape:{landmarks.shape}')
print(f'First 4 Landmarks:{landmarks[:4]}')