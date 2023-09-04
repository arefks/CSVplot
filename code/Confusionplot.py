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

cm = 1/2.54  # centimeters in inches

Path = r"Y:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\*\manual_slice_inspection"

p_anat = os.path.join(Path,"anat*.png")
p_func = os.path.join(Path,"func*.png")
p_struct = os.path.join(Path,"struct*.png")

anat_all = glob.glob(p_anat,recursive=True)
func_all = glob.glob(p_func,recursive=True)
struct_all = glob.glob(p_struct,recursive=True)

count_anat_all = len(anat_all)
count_func_all = len(func_all)
count_struct_all = len(struct_all)

#%% Expert


Path_gt = r"Y:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\*\validation_experienced"

pgt_anat = os.path.join(Path_gt,"anat*.png")
pgt_func = os.path.join(Path_gt,"func*.png")
pgt_struct = os.path.join(Path_gt,"struct*.png")

anatgt_all = glob.glob(pgt_anat,recursive=True)
funcgt_all = glob.glob(pgt_func,recursive=True)
structgt_all = glob.glob(pgt_struct,recursive=True)
# Update the lists to contain only the image names (file names)
anatgt_all = [os.path.basename(path) for path in anatgt_all]
funcgt_all = [os.path.basename(path) for path in funcgt_all]
structgt_all = [os.path.basename(path) for path in structgt_all]


countgt_anat_bad = len(anatgt_all)
coungt_func_bad = len(funcgt_all)
countgt_struct_bad = len(structgt_all)

countgt_anat_good = count_anat_all - countgt_anat_bad
coungt_func_good = count_func_all-coungt_func_bad
countgt_struct_good = count_struct_all-countgt_struct_bad






Path_votings = r"Y:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\*\votings.csv"

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
# Separate the new_df DataFrame based on specific prefixes
anatqc_all = new_df[new_df['corresponding_img'].str.startswith('anatomical')]['corresponding_img'].tolist()
structqc_all = new_df[new_df['corresponding_img'].str.startswith('structural')]['corresponding_img'].tolist()
funcqc_all = new_df[new_df['corresponding_img'].str.startswith('functional')]['corresponding_img'].tolist()

countqc_anat_bad = len(anatqc_all)
counqc_func_bad = len(structqc_all)
countqc_struct_bad = len(funcqc_all)

countqc_anat_good = count_anat_all - countqc_anat_bad
counqc_func_good = count_func_all-counqc_func_bad
countqc_struct_good = count_struct_all-countqc_struct_bad

anat_intersect_qc_gt = set(anatgt_all) & set(anatqc_all)
anat_TN = len(anat_intersect_qc_gt)
anat_FN = countqc_anat_bad - anat_TN
anat_FP = countgt_anat_bad - anat_TN
anat_TP = countgt_anat_good - anat_FN

anat_percent_TP = (anat_TP / countgt_anat_good)
anat_percent_FN = (1 - anat_percent_TP)
anat_percent_FP = (anat_FP /countgt_anat_bad)
anat_percent_TN = (1 - anat_percent_FP)


# Calculate precision
precision = anat_TP / (anat_TP + anat_FP)
# Calculate recall
recall = anat_TP / (anat_TP + anat_FN)

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the F1 score
print("F1 Score:", f1_score)

#%%
confusion_matrix = [[anat_percent_TP, anat_percent_FN],
                    [anat_percent_FP, anat_percent_TN]]

plt.figure(figsize=(6*cm, 6*cm), dpi=300)
# Create a heatmap using Seaborn
sns.set(font_scale=0.8)  # Adjust the font size
heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Greys',
                      annot_kws={"fontname": "Times New Roman"},
                      xticklabels=False, yticklabels=False, cbar=False)

# Customize the plot
plt.xlabel('AIDAqc', fontname='Times New Roman')
plt.ylabel('Expert User', fontname='Times New Roman')
plt.title('Anatomical data \n F1-Score: %.2f' % f1_score + "", fontname='Times New Roman',weight="bold")
plt.xticks([0.5, 1.5], ['good image', 'bad image'], fontname='Times New Roman')
plt.yticks([0.5, 1.5], ['good image', 'bad image'], fontname='Times New Roman')

plt.show()




# Show the plot
plt.show()
