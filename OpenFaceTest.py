# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:00:27 2022

@author: ntweat
"""

import pandas as pd
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib
import glob
import cv2 


openface_files =r"E:\Codes\Personal\2022\CVPR\OpenFace_Multi"
test_file = r"E:\Codes\Personal\2022\CVPR\3rd ABAW Annotations\Third ABAW Annotations\Valence_Arousal_Estimation_Challenge_test_set_release.txt"
out_dir = r"E:\Codes\Personal\2022\CVPR\output"
images = r"E:\Codes\Personal\2022\CVPR\OpenFace_Multi"
vids_b1  = r"E:\Codes\Personal\2022\CVPR\videos\batch1"
vids_b2 =  r"E:\Codes\Personal\2022\CVPR\videos\batch2"

included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']

model = joblib.load("new.sav")
vid_b1_lt = os.listdir(vids_b1)
vid_b2_lt = os.listdir(vids_b2)
test = pd.read_csv(test_file, header=None)

vid_b1_df = [x.split('.')[0] for x in vid_b1_lt]
vid_b2_df = [x.split('.')[0] for x in vid_b2_lt]

for idx, row in test.iterrows():
    print(row[0])
    if os.path.exists(os.path.join(out_dir,row[0]+'.txt')): continue
    vid = []
    vi_b1 = [v for v in vid_b1_lt if row[0].startswith(v.split('.')[0])]
    vi_b2 = [v for v in vid_b2_lt if row[0].startswith(v.split('.')[0])]
    if len(vi_b1)>0:
        vpath = os.path.join(vids_b1,vi_b1[0])
    else:
        vpath = os.path.join(vids_b2,vi_b2[0])
        
    video = cv2.VideoCapture(vpath)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    img_path = os.path.join(images, row[0]+"_aligned")
    
    frame_num = [fn.split(".")[0].replace('frame_det_00_','') for fn in os.listdir(img_path) if any(fn.endswith(ext) for ext in included_extensions)]
    
    file = os.path.join(openface_files, row[0]+".csv")
    opface = pd.read_csv(file)
    opface['fname'] = frame_num
    print(opface.head())
    
    ll = []
    sk = [-5,-5]
    iff = 0 
    for i in range(1, total+1):
        print(iff)
        print(i)
        if not int(opface.loc[iff]['fname']) == i:
            ll.append(sk)
            continue
        if int(opface.loc[iff][' success']) == 0:
            ll.append(sk)
            iff=iff+1
            continue
        
        feats = opface.loc[iff][[" gaze_0_x"," gaze_0_y"," gaze_0_z"," gaze_1_x"," gaze_1_y"," gaze_1_z", " AU01_r"," AU02_r",
                           " AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r",
                           " AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"," AU45_r"," AU01_c"," AU02_c"," AU04_c",
                           " AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c",
                           " AU20_c"," AU23_c"," AU25_c"," AU26_c"," AU28_c"," AU45_c"]].to_numpy().reshape(1, -1)
        
        vv = model.predict(feats)
        ll.append(vv[0].tolist())
        iff=iff+1
        
        
        
    df = pd.DataFrame(ll, columns=['valence','arousal'])
    df.to_csv(os.path.join(out_dir,row[0]+'.txt'),  index=False)
    #sys.exit(1)