
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

def calculate_confusion_matrix(path, Sequence, Voting_threshold):
    # Step 1: Accept an initial path as input and look for "voting.csv" files
    initial_path = path  # Replace with the actual path
    voting_csv_files = []
    for root, dirs, files in os.walk(initial_path):
        for file in files:
            if file.endswith("votings.csv"):
                voting_csv_files.append(os.path.join(root, file))
    
    # Step 2: Store all "voting.csv" files into a single Pandas DataFrame
    dataframes = []
    for file in voting_csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    aidaqc_voting = pd.concat(dataframes, ignore_index=True)
    
    # Step 3: Filter rows based on the "majority vote" column
    threshold = Voting_threshold
    aidaqc_voting_thresolded = aidaqc_voting[aidaqc_voting["Voting outliers (from 5)"] >= threshold]
    
    # Step 4: List all PNG images in the "manual_slice_inspection" folder
    manual_slice_inspection_path = os.path.join(initial_path, "manual_slice_inspection")
    manual_slice_images = [file for file in os.listdir(manual_slice_inspection_path) if file.endswith(".png")]
    
    # Create a DataFrame for "manual_slice_inspection"
    data = {"manual_slice_inspection": manual_slice_images}
    manual_voting = pd.DataFrame(data)
    
    # Step 5: Create columns for "validation_" folders
    validation_folders = [os.path.join(initial_path, folder) for folder in os.listdir(initial_path) if folder.startswith("validation_") and "Markus" not in folder]
    
    for folder in validation_folders:
        folder_name = os.path.basename(folder)
        folder_images = [file for file in os.listdir(folder) if file.endswith(".png")]
        
        # Create a column with 1 if the image is in the folder, 0 otherwise
        manual_voting[folder_name] = manual_voting["manual_slice_inspection"].apply(lambda x: 1 if x in folder_images else 0)
    
    # Now you have "VoteTemp" and "manual_voting" dataframes as described in steps 3 and 5 with the corrected code.
    manual_voting['MajorityVoteMR'] = manual_voting.iloc[:, 1:].sum(axis=1)
    manual_voting_thresholded = manual_voting[manual_voting["MajorityVoteMR"] >= threshold]
    S = Sequence
    #Separate the new_df DataFrame based on specific prefixes
    manual_voting_thresholded_afs = manual_voting_thresholded[manual_voting_thresholded['manual_slice_inspection'].str.startswith(S)]
    aidaqc_voting_thresolded_afs = aidaqc_voting_thresolded[aidaqc_voting_thresolded['corresponding_img'].str.startswith(S)] 
    
    
    temp_afs_all = manual_voting[manual_voting['manual_slice_inspection'].str.startswith(S)]
    count_afs_all = len(temp_afs_all)
    countqc_afs_bad = len(aidaqc_voting_thresolded_afs)
    countqc_afs_good = count_afs_all - countqc_afs_bad
    countgt_afs_bad = len(manual_voting_thresholded_afs)
    countgt_afs_good = len(temp_afs_all[temp_afs_all["MajorityVoteMR"] < threshold])
    
    
    afs_intersect_qc_gt_TP = set(aidaqc_voting_thresolded_afs['corresponding_img']) & set(manual_voting_thresholded_afs['manual_slice_inspection'])
    afs_intersect_qc_gt_FN = set(manual_voting_thresholded_afs['manual_slice_inspection']) & set(set(temp_afs_all['manual_slice_inspection'])-set(aidaqc_voting_thresolded_afs['corresponding_img']))
    afs_intersect_qc_gt_FP = set(set(temp_afs_all['manual_slice_inspection'])-set(manual_voting_thresholded_afs['manual_slice_inspection'])) & set(aidaqc_voting_thresolded_afs['corresponding_img'])
    afs_intersect_qc_gt_TN = set(set(temp_afs_all['manual_slice_inspection'])-set(manual_voting_thresholded_afs['manual_slice_inspection'])) &  set(set(temp_afs_all['manual_slice_inspection'])-set(aidaqc_voting_thresolded_afs['corresponding_img']))
    
    
    
    afs_TP = len(afs_intersect_qc_gt_TP)
    afs_FN = len(afs_intersect_qc_gt_FN)
    afs_FP = len(afs_intersect_qc_gt_FP)
    afs_TN = len(afs_intersect_qc_gt_TN)
    
    
    afs_percent_TP = (afs_TP / countgt_afs_bad)
    afs_percent_FN = (1 - afs_percent_TP)
    afs_percent_FP = (afs_FP /countgt_afs_good)
    afs_percent_TN = (1 - afs_percent_FP)
    
    
    # Calculate precision
    precision = afs_TP / (afs_TP + afs_FP)
    #print("precision"+str(precision))
    # Calculate recall
    recall = afs_TP / (afs_TP + afs_FN)
    #print("recall:"+str(recall))
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Print the F1 score
    #print("F1 Score:", f1_score)
    
    
    confusion_matrix = [[afs_percent_TP, afs_percent_FN],
                         [afs_percent_FP, afs_percent_TN]]
    return confusion_matrix, f1_score
    
    

    
    
    
    











