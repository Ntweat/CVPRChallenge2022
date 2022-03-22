# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:44:43 2022

@author: ntweat
"""

import os
import subprocess
import sys 

ip_path = r"E:\Codes\Personal\2022\CVPR\cropped\batch2 (1)\batch2"
op_path = r"E:\Codes\Personal\2022\CVPR\OpenFace_Multi"


for root, dirs, files in os.walk(ip_path):
    for dire in dirs:
        print(dire)
        #print(dire)
        #subprocess.run([r"E:\Codes\External\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FaceLandmarkVidMulti.exe", "-f", os.path.join(root, file), "-out_dir", op_path])
        subprocess.run([r"E:\Codes\External\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe", "-fdir", os.path.join(ip_path, dire), "-out_dir", op_path])
        #sys.exit(1)