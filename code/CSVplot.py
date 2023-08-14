import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import glob
import os
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
#%% Function to read CSV files and store them in a dictionary


def read_csv_files(files):
    data_dict = {}
    for ff,file in enumerate(files):
        df = pd.read_csv(file)    
        # Extract the data type from the file name (assuming the file name contains "anatomical", "structural", or "functional")
        data_type = "anatomical" if "anatomical" in file else ("structural" if "structural" in file else "functional")
        data_dict[ff] = {data_type:df}
            
    return data_dict


# Function for statistical comparison and plotting
def compare_and_plot(data, column_name, group_column):
    sns.set_palette("colorblind")  # Use colors for color-blind people
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_column, y=column_name, data=data)
    plt.xlabel(group_column)
    plt.ylabel(column_name)
    plt.title(f"Statistical Comparison of {column_name} by {group_column}")
    plt.tight_layout()
    plt.savefig(f"{column_name}_by_{group_column}_boxplot.png")
    plt.show()

#%% List of CSV files for each data type
Path = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_new_studies_04"

anatomical_files = glob.glob(os.path.join(Path,"**/*caculated_features_anatomical.csv"), recursive=True)
structural_files = glob.glob(os.path.join(Path,"**/*caculated_features_structural.csv"), recursive=True)
functional_files = glob.glob(os.path.join(Path,"**/*caculated_features_functional.csv"), recursive=True)

All_files = [anatomical_files,structural_files,functional_files]

# Read the CSV files and store them in dictionaries
anatomical_data = read_csv_files(anatomical_files)
structural_data = read_csv_files(structural_files)
functional_data = read_csv_files(functional_files)
All_Data = [anatomical_data,structural_data,functional_data]

All_type = ["anatomical","structural","functional"]
#%% Read files

features_to_compare = ["SNR Chang", "SNR Normal", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)"]

temp = pd.DataFrame()
Data_of_selected_feature = pd.DataFrame()
temp_data = pd.DataFrame()
cc = 0
for dd,data in All_Data:
    for feature in features_to_compare:
        for key in data:
            try:
                temp_data[feature] = data[key][All_type[dd]][feature]
            except KeyError:
                continue
            temp_data["Dataset"] = All_files[dd][cc].split(os.sep)[-3]
            cc = cc +1
            Data_of_selected_feature = pd.concat([Data_of_selected_feature, temp_data], ignore_index=True)
        
            #%% creating boxplots
            plt.figure(figsize=(7.09*2, 2.60*2),dpi=300)
            sns.set_style('white')
            sns.set(font='Times New Roman', font_scale=1.8,style=None)  # Set font to Times New Roman and font size to 9
            palette = 'Set2'
            ax = sns.violinplot(x="Dataset", y="SNR Normal", data=Data_of_selected_feature, hue="Dataset", dodge=False,
                                palette=palette,
                                scale="width", inner=None)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for violin in ax.collections:
                bbox = violin.get_paths()[0].get_extents()
                x0, y0, width, height = bbox.bounds
                violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
            
            sns.boxplot(x="Dataset", y="SNR Normal", data=Data_of_selected_feature, saturation=1, showfliers=False,
                        width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=2.8)
            old_len_collections = len(ax.collections)
            sns.stripplot(x="Dataset", y="SNR Normal", data=Data_of_selected_feature,size=4, hue="Dataset", palette=palette, dodge=False, ax=ax)
            for dots in ax.collections[old_len_collections:]:
                dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend_.remove()
            ax
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel('')
            #plt.savefig(os.path.join(Path,"FIGX.svg"), format='svg', bbox_inches='tight')
            plt.show()