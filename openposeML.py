# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:50:16 2022

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

openface_files =r"E:\Codes\Personal\2022\CVPR\OpenFace_Multi"
annot_files = r"E:\Codes\Personal\2022\CVPR\3rd ABAW Annotations\Third ABAW Annotations\VA_Estimation_Challenge"

train_files = os.path.join(annot_files, "Train_Set")
valid_files  = os.path.join(annot_files, "Validation_Set")


train_x = []
train_y = []

valid_x = []
valid_y =[]

'''
train_y['Val'] = []
train_y['Aro'] = []

valid_y['Val'] = []
valid_y['Aro'] = []
'''

ant_done = []
def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    '''
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    '''
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


for root, dirs, files in os.walk(train_files):
    for file in files:
        print(file)
        ant_file = os.path.join(root, file)
        opface_file = os.path.join(openface_files, file.replace(".txt", ".csv"))
        if not os.path.exists(ant_file) or not os.path.exists(opface_file): continue
        ant_df = pd.read_csv(ant_file)
        opface_df = pd.read_csv(opface_file)
        #print(opface_df[" face_id"].unique())
        #print(ant_df.head())
        #print(ant_df.mean())
        for idx, row in ant_df.iterrows():
            #print(row)
            try:
                feat_row = opface_df.loc[idx]
            except:
                continue
            if feat_row[" success"] ==0 or row["valence"]==-5 or row["arousal"]==-5: continue
            '''
            print(feat_row[[" gaze_0_x"," gaze_0_y"," gaze_0_z"," gaze_1_x"," gaze_1_y"," gaze_1_z", " AU01_r"," AU02_r",
                           " AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r",
                           " AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"," AU45_r"," AU01_c"," AU02_c"," AU04_c",
                           " AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c",
                           " AU20_c"," AU23_c"," AU25_c"," AU26_c"," AU28_c"," AU45_c"
]].to_numpy())
            '''
            feat_arr = feat_row[[" gaze_0_x"," gaze_0_y"," gaze_0_z"," gaze_1_x"," gaze_1_y"," gaze_1_z", " AU01_r"," AU02_r",
                           " AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r",
                           " AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"," AU45_r"," AU01_c"," AU02_c"," AU04_c",
                           " AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c",
                           " AU20_c"," AU23_c"," AU25_c"," AU26_c"," AU28_c"," AU45_c"
]].to_numpy()
            if np.isnan(feat_arr).any(): continue
            train_x.append(feat_arr)
            
            train_y.append([row["valence"],row["arousal"]])
            #train_y['Aro'].append(row["arousal"])
            #sys.exit(1)
        
        
        
       
        
        #sys.exit(1)
        
        
        
for root, dirs, files in os.walk(valid_files):
    for file in files:
        print(file)
        ant_file = os.path.join(root, file)
        opface_file = os.path.join(openface_files, file.replace(".txt", ".csv"))
        if not os.path.exists(ant_file) or not os.path.exists(opface_file): continue
        ant_df = pd.read_csv(ant_file)
        opface_df = pd.read_csv(opface_file)
        #print(opface_df[" face_id"].unique())
        #print(ant_df.head())
        #print(ant_df.mean())
        for idx, row in ant_df.iterrows():
            #print(row)
            try:
                feat_row = opface_df.loc[idx]
            except:
                continue
            #feat_row = opface_df.loc[idx]
            if feat_row[" success"] ==0 or row["valence"]==-5 or row["arousal"]==-5: continue

            feat_arr = feat_row[[" gaze_0_x"," gaze_0_y"," gaze_0_z"," gaze_1_x"," gaze_1_y"," gaze_1_z", " AU01_r"," AU02_r",
                           " AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r",
                           " AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"," AU45_r"," AU01_c"," AU02_c"," AU04_c",
                           " AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c",
                           " AU20_c"," AU23_c"," AU25_c"," AU26_c"," AU28_c"," AU45_c"
]].to_numpy()
            if np.isnan(feat_arr).any(): continue
            valid_x.append(feat_arr)
            valid_y.append([row["valence"],row["arousal"]])
            #valid_y['Aro'].append(row["arousal"])
            #sys.exit(1)
        
        
        
       
        
        #sys.exit(1)
        
        
        
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=250, random_state=0)
)
regr_multirf.fit(train_x, train_y)



hh = regr_multirf.predict(train_x)

print("Train")

print(concordance_correlation_coefficient(np.array(train_y)[:,0], hh[:,0]))
print(concordance_correlation_coefficient(np.array(train_y)[:,1], hh[:,1]))


hh = regr_multirf.predict(valid_x)

print("Valid")

print(concordance_correlation_coefficient(np.array(valid_y)[:,0], hh[:,0]))
print(concordance_correlation_coefficient(np.array(valid_y)[:,1], hh[:,1]))


joblib.dump(regr_multirf, "new.sav")