
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
from ConfusionMatrix_getter import *

#path = r"C:\Users\aswen\Desktop\Code\Validation\*/"
path = r"C:\Users\arefk\OneDrive\Desktop\Projects\Validation\*/"
List_folders = glob.glob(path)

Result = []

Sequence = ["anatomical","structural","functional"]
Voting_threshold = [1,2,3,4,5]
ErrorAll = []
for v in Voting_threshold:
    for L in List_folders:
        for S in Sequence:
            
            confusion_matrix,f1_score,kappa,Errors = calculate_confusion_matrix(L, S, v)
            Result.append({"Dataset":os.path.basename(os.path.abspath(L)),"Thresold":v,"Sequence":S,"Kappa":kappa,"F1_score":f1_score})
            if any(Errors):
                ErrorAll.append(Errors[0])
            # Create a heatmap using Seaborn
            cm = 1/2.54  # centimeters in inches
            
            plt.figure(figsize=(18*cm, 9*cm))
            fig, ax = plt.subplots(1,1, dpi=300,figsize=(18*cm, 9*cm))#,sharex="row",sharey="row")
            #fig.suptitle('Confusion matrix',fontname='Times New Roman')
                    
            
            sns.set(font_scale=0.8)  # Adjust the font size
            heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Greys',
                                  annot_kws={"fontname": "Times New Roman"},
                                  xticklabels=False, yticklabels=False, cbar=False)
            ax.set_xlabel('AIDAqc', fontname='Times New Roman')
            ax.set_ylabel("Manual Rater", fontname='Times New Roman')
            ax.set_title(S.capitalize()+" & "+ os.path.basename(os.path.abspath(L))  +'\n F1-score: %.2f' % f1_score + '  |  Kappa: %.2f' % kappa , fontname='Times New Roman', weight="bold")
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['bad', 'good'], fontname='Times New Roman')
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(['bad', 'good'], fontname='Times New Roman')
            plt.show()
ResultD = pd.DataFrame(Result)  
ResultD.to_excel(r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs\Kappa_and_F1Score_resultsTemp.xlsx")  
#%% All together

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
from statsmodels.stats.inter_rater import aggregate_raters, aggregate_raters
#from ConfusionMatrix_getter import *

path =  r"C:\Users\aswen\Desktop\Code\Validation\94_r_We"

Sequence = "anatomical"
Voting_threshold = 1
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
S = Sequence
# Step 5: Create columns for "validation_" folders
validation_folders = [os.path.join(initial_path, folder) for folder in os.listdir(initial_path) if folder.startswith("validation_") and "Markus" not in folder]

for folder in validation_folders:
    folder_name = os.path.basename(folder)
    folder_images = [file for file in os.listdir(folder) if file.endswith(".png")]
    
    # Create a column with 1 if the image is in the folder, 0 otherwise
    manual_voting[folder_name] = manual_voting["manual_slice_inspection"].apply(lambda x: 1 if x in folder_images else 0)

# Now you have "VoteTemp" and "manual_voting" dataframes as described in steps 3 and 5 with the corrected code.
manual_voting['MajorityVoteMR'] = manual_voting.iloc[:, 1:].sum(axis=1)
manual_voting_afs = manual_voting[manual_voting['manual_slice_inspection'].str.startswith(S)]



# Create a new DataFrame containing only the relevant columns
columns_to_include = [col for col in manual_voting_afs.columns if col != "manual_slice_inspection" and col != "MajorityVoteMR"]
ratings_data = manual_voting_afs[columns_to_include]

# Convert the ratings data to a format expected by fleiss_kappa
ratings_matrix,d = aggregate_raters(ratings_data.values, n_cat=None) 

# Calculate Fleiss' Kappa
kappa = fleiss_kappa(ratings_matrix,method='fleiss')



manual_voting_thresholded = manual_voting[manual_voting["MajorityVoteMR"] >= threshold]

#Separate the new_df DataFrame based on specific prefixes
manual_voting_thresholded_afs = manual_voting_thresholded[manual_voting_thresholded['manual_slice_inspection'].str.startswith(S)]
aidaqc_voting_thresolded_afs = aidaqc_voting_thresolded[aidaqc_voting_thresolded['corresponding_img'].str.startswith(S)] 


temp_afs_all = manual_voting[manual_voting['manual_slice_inspection'].str.startswith(S)]
count_afs_all = len(temp_afs_all)
countqc_afs_bad = len(aidaqc_voting_thresolded_afs)
countqc_afs_good = count_afs_all - countqc_afs_bad
countgt_afs_bad = len(manual_voting_thresholded_afs)
countgt_afs_good = count_afs_all - countgt_afs_bad


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
confusion_matrix, f1_score

    
# Create a heatmap using Seaborn
cm = 1/2.54  # centimeters in inches

plt.figure(figsize=(18*cm, 9*cm))
fig, ax = plt.subplots(1,1, dpi=300,figsize=(18*cm, 9*cm))#,sharex="row",sharey="row")
#fig.suptitle('Confusion matrix',fontname='Times New Roman')
        

sns.set(font_scale=0.8)  # Adjust the font size
heatmap = sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Greys',
                      annot_kws={"fontname": "Times New Roman"},
                      xticklabels=False, yticklabels=False, cbar=False)
ax.set_xlabel('AIDAqc', fontname='Times New Roman')
ax.set_ylabel("Manual Rater", fontname='Times New Roman')
ax.set_title(Sequence.capitalize()+'\n F1-score: %.2f' % f1_score + "", fontname='Times New Roman', weight="bold")
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['bad', 'good'], fontname='Times New Roman')
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(['bad', 'good'], fontname='Times New Roman')


    
    











