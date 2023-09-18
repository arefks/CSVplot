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

#%function

def calculate_and_display_fleiss_kappa(data):
    kappa, _ = fleiss_kappa(data)
    print("Fleiss' Kappa:", kappa)
    return kappa

#%% Expert

Path_votings_chang = r"C:\Users\aswen\Documents\Data\2023_Kalantari_AIDAqc\outputs\validation\QC_Chang\*\votings.csv"

All_csv_votings = glob.glob(Path_votings_chang)

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

chang_df = pd.DataFrame({"corresponding_img": img_values})
###
###

Path_votings_standard = r"C:\Users\aswen\Documents\Data\2023_Kalantari_AIDAqc\outputs\validation\QC_standard\*\votings.csv"

All_csv_votings = glob.glob(Path_votings_standard)

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

standard_df = pd.DataFrame({"corresponding_img": img_values})
