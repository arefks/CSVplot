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
text=[]
def calculate_and_display_fleiss_kappa(data):
    kappa, _ = fleiss_kappa(data)
    print("Fleiss' Kappa:", kappa)
    return kappa

def read_csv_files(files):
    data_dict = {}
    for ff,file in enumerate(files):
        df = pd.read_csv(file)    
        data_dict[ff] = df[df["Voting outliers (from 5)"]>0] 
            
    return data_dict
#%% Expert

Path_votings = r"C:\Users\aswen\Desktop\validation\*\votings.csv"
#P = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\7_m_Br\votings.csv"


All_csv_votings = glob.glob(Path_votings)

for aa in All_csv_votings:
    
    temp = glob.glob(aa)
    All_values = read_csv_files(temp)  
    print(temp)
    
    img_values = []
    
    for df_name, df in All_values.items():
        if "corresponding_img" in df.columns:
            img_values.extend(df["corresponding_img"].tolist())
    
    new_df = pd.DataFrame({"corresponding_img": img_values})
    
    MU_strings = ["beginner","experienced","expert"]
    Sequence_type = ["anatomical","functional","structural"]
    cc=0
    cm = 1/2.54  # centimeters in inches
    
    plt.figure(figsize=(18*cm, 9*cm))
    fig, ax = plt.subplots(3,3, dpi=300,figsize=(12*cm, 12*cm))#,sharex="row",sharey="row")
    #fig.suptitle('Confusion matrix',fontname='Times New Roman')
            
    Matric_sequneces = []
    for ss,S in enumerate(Sequence_type):
        for mu,MU in enumerate(MU_strings):
            
            #Path = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\94_m_As\manual_slice_inspection"
            Path = os.path.join(os.path.dirname(temp[0]), "manual_slice_inspection")
            
            
            p_afs = os.path.join(Path,S+"*.png")
            
            afs_all = glob.glob(p_afs,recursive=True)
            afs_all = [os.path.basename(path) for path in afs_all]
            if not afs_all:
                print("fuck that " + MU + "and " + S)
                continue
            
            
            count_afs_all = len(afs_all)
            
    
            #Path_gt = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\94_m_As\validation_" + MU
            Path_gt = os.path.join(os.path.dirname(temp[0]),("validation_" + MU))
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
    # =============================================================================
    #         afs_TN = len(afs_intersect_qc_gt)
    #         afs_FN = countqc_afs_bad - afs_TN
    #         afs_FP = countgt_afs_bad - afs_TN
    #         afs_TP = countgt_afs_good - afs_FN
    # =============================================================================
            afs_TP = len(afs_intersect_qc_gt)
            afs_FN = countgt_afs_bad - afs_TP
            afs_FP = countqc_afs_bad - afs_TP
            afs_TN = countgt_afs_good - afs_FP
            
    # =============================================================================
            afs_percent_TP = (afs_TP / countgt_afs_bad)
            afs_percent_FN = (1 - afs_percent_TP)
            afs_percent_FP = (afs_FP /countgt_afs_good)
            afs_percent_TN = (1 - afs_percent_FP)
    # =============================================================================
            
            
            # Calculate precision
            precision = afs_TP / (afs_TP + afs_FP)

            # Calculate recall
            recall = afs_TP / (afs_TP + afs_FN)

            
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # Print the F1 score
            print("F1 Score:", f1_score)
    
            
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
            
            
            
            Vec = [0 if item in afsgt_all else 1 for item in afs_all]
            if not Vec:
                print("fuck that precsion and recall " + MU + "and " + S)
                continue
            
            Matric_sequneces.append(Vec)
            
    
        Vec_qc = [0 if item in afsqc_all else 1 for item in afs_all]   
        Matric_sequneces.append(Vec_qc)   
            
            
            
            
            
            
    # Show the plot
    #plt.subplots_adjust(left=0.00, right=1, top=1, bottom=0.00, wspace=0.3, hspace=1.8)
    plt.tight_layout()
    plt.show()
    
    
    
    try:
        Stacked_anat = np.stack(Matric_sequneces[0:4],axis=0).transpose()
        Stacked_func = np.stack(Matric_sequneces[4:8],axis=0).transpose()
        Stacked_struct = np.stack(Matric_sequneces[8:12],axis=0).transpose()
        
        Stacked_anat_bool = Stacked_anat.astype(bool)
        Stacked_func_bool = Stacked_func.astype(bool)
        Stacked_struct_bool = Stacked_struct.astype(bool)
    except ValueError:
        print("fuck that precsion and recall " + MU + "and " + S)
        continue
    
    # Define column labels
    column_labels = ["beginner", "experienced", "expert", "aidaqc"]
    
    # Create pandas DataFrames
    try:
        df_anat = pd.DataFrame(Stacked_anat_bool, columns=column_labels)
        df_func = pd.DataFrame(Stacked_func_bool, columns=column_labels)
        df_struct = pd.DataFrame(Stacked_struct_bool, columns=column_labels)
    except ValueError:
        print("fuck that precsion and recall " + MU + "and " + S)
        continue
    
        
    from sklearn.metrics import cohen_kappa_score
    
    # Create a list of DataFrames
    dataframes = [df_anat, df_func, df_struct]
    
    # Loop through the DataFrames
    for i, df in enumerate(dataframes):
        print(f"DataFrame {i + 1}:")
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    kappa_score = cohen_kappa_score(df[col1], df[col2])
                    print(f"Cohen's Kappa Score for {Sequence_type[i]} and between {col1} and {col2}: {kappa_score:.4f}")
                    text.append(f"Cohen's Kappa Score for {Sequence_type[i]} and {aa} between {col1} and {col2}: {kappa_score:.4f}")
        print()
    
    
