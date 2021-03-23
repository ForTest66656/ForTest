import os
from matplotlib.pyplot import annotate

import pandas as pd
import numpy as np

landmarks_frame = pd.read_csv('data/300w/face_landmarks_300w_train.csv')
landmarks_frame_test = pd.read_csv('data/300w/face_landmarks_300w_valid.csv')
df_train = dict()
df_test = dict()

with open('data/300w/images/helen/train.txt') as f:
    trainset = f.readlines()
trainset = [x.strip() for x in trainset]

with open('data/300w/images/helen/test.txt') as f:
    testset = f.readlines()
testset = [x.strip() for x in testset]

for idx in range(len(landmarks_frame)):
    name = landmarks_frame.iloc[idx, 0]
    if 'helen' not in name:
        continue
    fn = os.path.split(name)[1]
    print(name)
    annotation_idx = trainset.index(fn[:-4])
    with open('data/300w/images/helen/annotation/{}.txt'.format(annotation_idx + 1)) as f:
        pts = f.readlines()
        pts = [x.strip() for x in pts]
        pts = pts[1:]
        pts = [x.split(',') for x in pts]
        flat_pts = [float(item) for sublist in pts for item in sublist]

    df_train[name] = list(landmarks_frame.iloc[idx, 0:4]) + flat_pts

for idx in range(len(landmarks_frame_test)):
    name = landmarks_frame_test.iloc[idx, 0]
    if 'helen' not in name:
        continue
    fn = os.path.split(name)[1]
    annotation_idx = testset.index(fn[:-4]) + 2000
    with open('data/300w/images/helen/annotation/{}.txt'.format(annotation_idx + 1)) as f:
        pts = f.readlines()
        pts = [x.strip() for x in pts]
        pts = pts[1:]
        pts = [x.split(',') for x in pts]
        flat_pts = [float(item) for sublist in pts for item in sublist]

    df_test[name] = list(landmarks_frame_test.iloc[idx, 0:4]) + flat_pts

df_train = pd.DataFrame.from_dict(df_train)        
df_test = pd.DataFrame.from_dict(df_test)
df_train.to_csv('data/300w/images/helen/face_landmarks_helen_train.csv') 
df_test.to_csv('data/300w/images/helen/face_landmarks_helen_test.csv')

print('Finish')