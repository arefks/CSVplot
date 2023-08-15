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

#% List of CSV files for each data type
Path = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_new_studies_04\validation"

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
#% data statistisc figure 7

features_to_compare = ["SNR Chang", "SNR Normal", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)"]


for dd,data in enumerate(All_Data):
    for feature in features_to_compare:
        cc = 0
        temp = pd.DataFrame()
        Data_of_selected_feature = pd.DataFrame()
        temp_data = pd.DataFrame()

        for key in data:
            try:
                temp_data[feature] = data[key][All_type[dd]][feature]
            except KeyError:
                continue
            temp_data["Dataset"] = All_files[dd][cc].split(os.sep)[-3]
            cc = cc +1
            Data_of_selected_feature = pd.concat([Data_of_selected_feature, temp_data], ignore_index=True)
        if not Data_of_selected_feature.empty:
            # creating boxplots
            plt.figure(figsize=(7.09*2, 2.60*2),dpi=300)
            sns.set_style('ticks')
            sns.set(font='Times New Roman', font_scale=1.8,style=None)  # Set font to Times New Roman and font size to 9
            palette = 'Set2'
            ax = sns.violinplot(x="Dataset", y=feature, data=Data_of_selected_feature, hue="Dataset", dodge=False,
                                palette=palette,
                                scale="width", inner=None)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for violin in ax.collections:
                bbox = violin.get_paths()[0].get_extents()
                x0, y0, width, height = bbox.bounds
                violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
            
            sns.boxplot(x="Dataset", y=feature, data=Data_of_selected_feature, saturation=1, showfliers=False,
                        width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=2.8)
            old_len_collections = len(ax.collections)
            sns.stripplot(x="Dataset", y=feature, data=Data_of_selected_feature,size=4, hue="Dataset", palette=palette, dodge=False, ax=ax)
            for dots in ax.collections[old_len_collections:]:
                dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend_.remove()
            ax
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel('')
            ax.set_title(All_type[dd])
           
            ax.xaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=1)
            ax.xaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)

            ax.yaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=1)
            ax.yaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)        

            plt.savefig(os.path.join(Path,feature+"_"+All_type[dd]+".svg"), format='svg', bbox_inches='tight')
            plt.show()
#%% Data statistics figure 6


p_address = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\docs\AIDAqc_Testdaten.csv"
p_save = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_new_studies_04\validation"

# Load the CSV data into a pandas DataFrame
data = pd.read_csv(p_address)
# Clean up the 'Scanner' column to extract the field strength (including decimals)

def extract_field_strength(scanner):
    match = re.search(r'(\d+(\.\d+)?)T', scanner)
    if match:
        return float(match.group(1))
    else:
        return None

data['Scanner'] = data['Scanner'].apply(extract_field_strength)

# =============================================================================
# # 1. Sample size distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(data=data, x='Sample (n)', bins=30, kde=True)
# plt.xlabel('Sample Size')
# plt.ylabel('Frequency')
# plt.title('Sample Size Distribution')
# plt.show()
# =============================================================================
sns.set_palette("colorblind")  # Use colors for color-blind people
sns.set(font='Times New Roman', font_scale=0.65,style=None,rc={"font.weight": "bold"})  # Set font to Times New Roman and font size to 9
# 2. Species pie chart
species_counts = data['Species'].value_counts()
plt.figure(figsize=(2.77, 2.77), dpi=300)
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140,pctdistance=0.85)


sns.set(font='Times New Roman', font_scale=0.9,style=None,rc={"font.weight": "bold"})  # Set font to Times New Roman and font size to 9

plt.title('Species Distribution')

plt.savefig(p_save +'Species Distribution' +".svg", format='svg', bbox_inches='tight')
plt.show()
# 3. Field strength pie chart
scanner_counts = data['Scanner'].value_counts()
plt.figure(figsize=(2.77, 2.77), dpi=300)
sns.set(font='Times New Roman', font_scale=0.7,style=None,rc={"font.weight": "bold"})  # Set font to Times New Roman and font size to 9
plt.pie(scanner_counts, labels=scanner_counts.index, autopct='%1.1f%%', startangle=140,pctdistance=0.8)
sns.set(font='Times New Roman', font_scale=0.9,style=None,rc={"font.weight": "bold"})  # Set font to Times New Roman and font size to 9

plt.title('Field Strength Distribution (T)')

plt.savefig(p_save +'Field Strength Distribution' +".svg", format='svg', bbox_inches='tight')
plt.show()
# 4. Sequence type pie chart
# Split the combined sequences by commas and create a new DataFrame
sequences_data = data['Sequences'].str.split(', ', expand=True)

# Reshape the sequences DataFrame to long format
sequences_melted = sequences_data.melt(value_name='Sequence').dropna()['Sequence']

# Create the sequence type pie chart
plt.figure(figsize=(2.77, 2.77), dpi=300)
sequence_counts = sequences_melted.value_counts()
plt.pie(sequence_counts, labels=sequence_counts.index, autopct='%1.1f%%', startangle=140,pctdistance=0.7)
plt.title('Sequence Type Distribution')

plt.savefig(p_save +'Sequence Type Distribution'+".svg", format='svg', bbox_inches='tight')
plt.show()
# 5. Data format pie chart
format_counts = data['Data format'].value_counts()
plt.figure(figsize=(2.77, 2.77), dpi=300)
plt.pie(format_counts, labels=format_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Data Format Distribution')

plt.savefig(p_save +"Data Format Distribution" +".svg", format='svg', bbox_inches='tight')
plt.show()













































            
            
            
            
            
            
            
            