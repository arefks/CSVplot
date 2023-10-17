
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
    Errors = []
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
    S = Sequence
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
    manual_voting_afs = manual_voting[manual_voting['manual_slice_inspection'].str.startswith(S)] 
    
    # Create a new DataFrame containing only the relevant columns
    columns_to_include = [col for col in manual_voting_afs.columns if col != "manual_slice_inspection" and col != "MajorityVoteMR"]
    ratings_data = manual_voting_afs[columns_to_include]

    try:
        # Convert the ratings data to a format expected by fleiss_kappa
        ratings_matrix,d = aggregate_raters(ratings_data.values, n_cat=None) 
    
        # Calculate Fleiss' Kappa
        kappa = fleiss_kappa(ratings_matrix,method='fleiss')
    except ValueError:
        print("\n "+"value Error, no kappa could be calculated for:")
        print(path + Sequence)
        Errors.append(path + Sequence)
        kappa = 0

    
    
    manual_voting['MajorityVoteMR'] = manual_voting.iloc[:, 1:].sum(axis=1)
    manual_voting_thresholded = manual_voting[manual_voting["MajorityVoteMR"] >= threshold]
    
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
    
    if countgt_afs_bad == 0:
        print("\n "+"countgt_afs_bad was zero for:")
        print(path + Sequence)
        afs_percent_TP = 0
        Errors.append(path + Sequence)
    else:
        afs_percent_TP = (afs_TP / countgt_afs_bad)
    
    afs_percent_FN = (1 - afs_percent_TP)
    
    if countgt_afs_good == 0:
        afs_percent_FP = 0
        print("\n "+"countgt_afs_good was zero for:")
        print(path + Sequence)
        Errors.append(path + Sequence)
    else:
        afs_percent_FP = (afs_FP /countgt_afs_good)
    
    afs_percent_TN = (1 - afs_percent_FP)
    
    
    if afs_TP + afs_FP == 0:
        print("\n "+"afs_TP + afs_FP was zero for:")
        print(path + Sequence)
        precision = 0
        Errors.append(path + Sequence)
    else:
        precision = afs_TP / (afs_TP + afs_FP)

    # Calculate recall
    if afs_TP + afs_FN == 0:
        print("\n "+"afs_TP + afs_FN was zero for:")
        print(path + Sequence)
        recall = 0
        Errors.append(path + Sequence)
    else:
        recall = afs_TP / (afs_TP + afs_FN)

    # Calculate F1 score
    if precision + recall == 0:
        print("\n "+"precision + recall was zero for:")
        print(path + Sequence)
        f1_score = 0
        Errors.append(path + Sequence)
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)    
    
    
    confusion_matrix = [[afs_percent_TP, afs_percent_FN],
                         [afs_percent_FP, afs_percent_TN]]
    return confusion_matrix, f1_score, kappa, np.unique(Errors)
    
    

    
    
    
    











