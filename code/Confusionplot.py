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

#%


#%% Expert

Path_votings = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\*\votings.csv"

All_csv_votings = glob.glob(Path_votings)

def read_csv_files(files):
    data_dict = {}
    for ff,file in enumerate(files):
        df = pd.read_csv(file)    
        data_dict[ff] = df 
            
    return data_dict

All_values = read_csv_files(All_csv_votings)

img_values = []

for df_name, df in All_values.items():
    if "corresponding_img" in df.columns:
        img_values.extend(df["corresponding_img"].tolist())

new_df = pd.DataFrame({"corresponding_img": img_values})

MU_strings = ["beginner","experienced","expert"]
Sequence_type = ["anatomical","functional","structural"]
cc=0
cm = 1/2.54  # centimeters in inches

plt.figure(figsize=(18*cm, 18*cm), dpi=300)
fig, ax = plt.subplots(3,3)
#fig.suptitle('Confusion matrix',fontname='Times New Roman')
        

for mu,MU in enumerate(MU_strings):
    for ss,S in enumerate(Sequence_type):
        
        Path = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\*\manual_slice_inspection"
        
        p_afs = os.path.join(Path,S+"*.png")
        
        afs_all = glob.glob(p_afs,recursive=True)
        
        count_afs_all = len(afs_all)
        

        Path_gt = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\*\validation_" + MU
        pgt_afs = os.path.join(Path_gt, S+"*.png")
        
        afsgt_all = glob.glob(pgt_afs,recursive=True)
        
        afsgt_all = [os.path.basename(path) for path in afsgt_all]
        
        countgt_afs_bad = len(afsgt_all)
        
        countgt_afs_good = count_afs_all - countgt_afs_bad
        
        
        
        
        
        # Separate the new_df DataFrame based on specific prefixes
        afsqc_all = new_df[new_df['corresponding_img'].str.startswith(S)]['corresponding_img'].tolist()
        
        countqc_afs_bad = len(afsqc_all)
        countqc_afs_good = count_afs_all - countqc_afs_bad
        
        
        afs_intersect_qc_gt = set(afsgt_all) & set(afsqc_all)
        afs_TN = len(afs_intersect_qc_gt)
        afs_FN = countqc_afs_bad - afs_TN
        afs_FP = countgt_afs_bad - afs_TN
        afs_TP = countgt_afs_good - afs_FN
        
        afs_percent_TP = (afs_TP / countgt_afs_good)
        afs_percent_FN = (1 - afs_percent_TP)
        afs_percent_FP = (afs_FP /countgt_afs_bad)
        afs_percent_TN = (1 - afs_percent_FP)
        
        
        # Calculate precision
        precision = afs_TP / (afs_TP + afs_FP)
        # Calculate recall
        recall = afs_TP / (afs_TP + afs_FN)
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Print the F1 score
        print("F1 Score:", f1_score)

#%
        confusion_matrix = [[afs_percent_TP, afs_percent_FN],
                            [afs_percent_FP, afs_percent_TN]]
        
        
        # Create a heatmap using Seaborn
        sns.set(font_scale=0.8)  # Adjust the font size
        heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Greys',
                              annot_kws={"fontname": "Times New Roman"},
                              xticklabels=False, yticklabels=False, cbar=False,ax=ax[ss,mu])
        ax[ss, mu].set_xlabel('AIDAqc', fontname='Times New Roman')
        ax[ss, mu].set_ylabel(MU, fontname='Times New Roman')
        ax[ss, mu].set_title(S.capitalize()+'\n F1-score: %.2f' % f1_score + "", fontname='Times New Roman', weight="bold")
        ax[ss, mu].set_xticks([0.5, 1.5])
        ax[ss, mu].set_xticklabels(['good', 'bad'], fontname='Times New Roman')
        ax[ss, mu].set_yticks([0.5, 1.5])
        ax[ss, mu].set_yticklabels(['good', 'bad'], fontname='Times New Roman',rotation=90)
# Show the plot
plt.subplots_adjust(left=0.00, right=1, top=1, bottom=0.00, wspace=0.3, hspace=1.1)
#plt.tight_layout()
plt.show()
