# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:57:59 2023

@author: arefks
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import glob
import os
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import re
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.inter_rater import aggregate_raters
from shutil import copyfile
#%function

def calculate_and_display_fleiss_kappa(data):
    kappa, _ = fleiss_kappa(data)
    print("Fleiss' Kappa:", kappa)
    return kappa

#%% Expert

Path_votings = r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\test\*\votings.csv"

All_csv_votings = glob.glob(Path_votings)

def read_csv_files(files):
    data_dict = {}
    for ff,file in enumerate(files):
        df = pd.read_csv(file)    
        data_dict[ff] = df[df["Voting outliers (from 5)"]>0] 
            
    return data_dict

All_values = read_csv_files(All_csv_votings)

img_values = []

for df_name, df in All_values.items():
    if "corresponding_img" in df.columns:
        img_values.extend(df["corresponding_img"].tolist())

new_df = pd.DataFrame({"corresponding_img": img_values})


PP =  r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\test"
    
for tt in new_df["corresponding_img"]: 
    
    String = os.path.join(PP,"**",tt)
    PathIm = glob.glob(String,recursive=True)
    temp = os.path.dirname(PathIm[0])
    Path_study = os.path.dirname(temp)
    Path_new_folder = os.path.join(Path_study,"votedImages")
    PathIm_new = os.path.join(Path_new_folder,tt)
    if not os.path.exists(Path_new_folder):
        os.mkdir(Path_new_folder)
        
    copyfile(PathIm[0],PathIm_new)
    print(String)





