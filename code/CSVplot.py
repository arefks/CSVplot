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

#% Function to read CSV files and store them in a dictionary

cm = 1/2.54  # centimeters in inches
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
    sns.set_palette("Set2")  # Use colors for color-blind people
    plt.figure(figsize=(9*cm, 6*cm))
    sns.boxplot(x=group_column, y=column_name, data=data)
    plt.xlabel(group_column)
    plt.ylabel(column_name)
    plt.title(f"Statistical Comparison of {column_name} by {group_column}")
    plt.tight_layout()
    #plt.savefig(f"{column_name}_by_{group_column}_boxplot.png")
    plt.show()

#% List of CSV files for each data type
Path = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_Final\validation"

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
            Data_of_selected_feature['sort'] = Data_of_selected_feature['Dataset'].str.extract('(\d+)', expand=True).astype(int)
            Data_of_selected_feature = Data_of_selected_feature.sort_values('sort')
            
            if feature == "SNR Normal":
                Data_of_selected_feature.rename(columns={"SNR Normal": "SNR-Standard (dB)"}, inplace=True)
                feature = "SNR-Standard (dB)"
            if feature == "SNR Chang":
                Data_of_selected_feature.rename(columns={"SNR Chang": "SNR-Chang (dB)"}, inplace=True)
                feature = "SNR-Chang (dB)"    
            elif feature == "tSNR (Averaged Brain ROI)":
                Data_of_selected_feature.rename(columns={"tSNR (Averaged Brain ROI)": "tSNR (dB)"}, inplace=True)
                feature = "tSNR (dB)"
            elif feature == "Displacement factor (std of Mutual information)":
                Data_of_selected_feature.rename(columns={"Displacement factor (std of Mutual information)": "Motion severity (A.U)"}, inplace=True)
                feature = "Motion severity (A.U)"
            
            
            
            
            #Data_of_selected_feature = Data_of_selected_feature.sort_values("Dataset",ascending=False)
            # creating boxplots
            plt.figure(figsize=(9.59*cm,3.527*cm),dpi=300)
            sns.set_style('ticks')
            sns.set(font='Times New Roman', font_scale=0.9,style=None)  # Set font to Times New Roman and font size to 9
            palette = 'Set2'
            ax = sns.violinplot(x="Dataset", y=feature, data=Data_of_selected_feature, hue="Dataset", dodge=False,
                                palette=palette,
                                scale="width", inner=None,linewidth=1)
            patches = ax.patches
            #legend_colors = [patch.get_facecolor() for patch in patches[:]]

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for violin in ax.collections:
                bbox = violin.get_paths()[0].get_extents()
                x0, y0, width, height = bbox.bounds
                violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
            
            sns.boxplot(x="Dataset", y=feature, data=Data_of_selected_feature, saturation=1, showfliers=False,
                        width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=1)
            old_len_collections = len(ax.collections)
            sns.stripplot(x="Dataset", y=feature, data=Data_of_selected_feature,size=1.1, hue="Dataset", palette=palette, dodge=False, ax=ax)
            for dots in ax.collections[old_len_collections:]:
                dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend_.remove()
            ax
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45,fontsize=8)
# =============================================================================
#             for label, color in zip(ax.get_xticklabels(), legend_colors):
#                 label.set_color(color)
# =============================================================================
            ax.set_xlabel('')
            ax.set_title(All_type[dd].capitalize(),weight='bold',fontsize=9)
            y_label = ax.set_ylabel(ax.get_ylabel(), fontweight='bold',fontsize=9)

# =============================================================================
            ax.xaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=0.5)
            ax.xaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)
# 
            ax.yaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=0.5)
            ax.yaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)        
# =============================================================================
            ax.spines['top'].set_linewidth(0.5)  # Top border
            ax.spines['right'].set_linewidth(0.5)  # Right border
            ax.spines['bottom'].set_linewidth(0.5)  # Bottom border
            ax.spines['left'].set_linewidth(0.5)  # Left border
            ax.tick_params(axis='both', which='both', width=0.5,color='gray',length=2)
            plt.xticks(ha='right')
            #plt.savefig(os.path.join(Path,feature+"_"+All_type[dd]+".svg"), format='svg', bbox_inches='tight',transparent=False)
            plt.show()
#%% Data statistics figure 6


p_address = r"Y:\Student_projects\14_Aref_Kalantari_2021\Projects\QC\AIDAqc_Testdaten.csv"
p_save = r"Y:\Student_projects\14_Aref_Kalantari_2021\Projects\QC\PieChart"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set the color palette to 'Set2'
sns.set(font='Times New Roman', font_scale=0.8, palette="pastel")

# Load the CSV data into a pandas DataFrame
data = pd.read_csv(p_address)
# Clean up the 'Scanner' column to extract the field strength (including decimals)

def extract_field_strength(scanner):
    match = re.search(r'(\d+(\.\d+)?)T', scanner)
    if match:
        return f"{match.group(1)}T"
    else:
        return None

data['Scanner'] = data['Scanner'].apply(extract_field_strength)
cm = 1/2.54  # centimeters in inches
# Set up the figure and axes using subplots
fig, axes = plt.subplots(1, 4, figsize=(22.62*cm,100*cm), dpi=300)

# Species pie chart
species_counts = data['Species'].value_counts()
axes[0].pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=180, pctdistance=0.75)
axes[0].set_title('(a) Species Distribution', weight='bold', fontsize=8)

# Field strength pie chart
scanner_counts = data['Scanner'].value_counts()
axes[1].pie(scanner_counts, labels=scanner_counts.index, autopct='%1.1f%%', startangle=180, pctdistance=0.70)
axes[1].set_title('(b) Field Strength Distribution', weight='bold', fontsize=8)

# Sequence type pie chart
sequences_data = data['Sequences'].str.split(', ', expand=True)
sequences_melted = sequences_data.melt(value_name='Sequence').dropna()['Sequence']
sequence_counts = sequences_melted.value_counts()
axes[2].pie(sequence_counts, labels=sequence_counts.index, autopct='%1.1f%%', startangle=180, pctdistance=0.65)
axes[2].set_title('(c) Sequence Type Distribution', weight='bold', fontsize=8)

# Data format pie chart
format_counts = data['Data format'].value_counts()
axes[3].pie(format_counts, labels=format_counts.index, autopct='%1.1f%%', startangle=180)
axes[3].set_title('(d) Data Format Distribution', weight='bold', fontsize=8)

# Turn off axes for all subplots
for ax in axes:
    ax.axis('off')

# Adjust layout and save the figure
#plt.tight_layout()
#plt.savefig(p_save + "All_Pie_Charts" + ".svg", format='svg', bbox_inches='tight')
plt.show()

#%% SNR Chang vs Standard plot for RT and Cryp

p_address = r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\94_m_As\calculated_features\caculated_features_anatomical.csv"
p_address2= r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\94c_m_As\calculated_features\caculated_features_anatomical.csv"
#p_save = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\ChangVSStandard"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = 1/2.54  # centimeters in inches
# Set the color palette to 'Set2'

plt.figure(figsize=(18*cm/3,7.30*cm),dpi=300)
# Load the CSV data into a pandas DataFrame
data = pd.read_csv(p_address)
data2 = pd.read_csv(p_address2)



chang = data["SNR Chang"].to_frame().rename(columns={"SNR Chang": "SNR"})
normal = data["SNR Normal"].to_frame().rename(columns={"SNR Normal": "SNR"})
chang["type"] = "chang_RT"
normal["type"] = "normal_RT"



chang2 = data2["SNR Chang"].to_frame().rename(columns={"SNR Chang": "SNR"})
normal2 = data2["SNR Normal"].to_frame().rename(columns={"SNR Normal": "SNR"})
chang2["type"] = "chang_Cryo"
normal2["type"] = "normal_Cryo"




Data_merged = data[["SNR Chang","SNR Normal"]]
Data_merged.rename(columns={"SNR Normal":"SNR-standard"})
Data_merged[["SNR Chang","SNR Normal"]] = Data_merged[["SNR Chang","SNR Normal"]].apply(lambda x: np.power((x/20),10))

Data_merged2 = data2[["SNR Chang","SNR Normal"]]
Data_merged2.rename(columns={"SNR Normal":"SNR-standard"})
Data_merged2[["SNR Chang","SNR Normal"]] = Data_merged2[["SNR Chang","SNR Normal"]].apply(lambda x: np.power((x/20),10))


SNR = pd.concat([chang, normal,chang2,normal2])
SNR["SNR"] = SNR["SNR"].apply(lambda x: np.power(x/20,10))

tips = SNR
x = "type"
y = "SNR"
sns.set_style('ticks')
# Create circular marker style for left side
circle_marker = {"marker": "o", "markerfacecolor": "black", "markersize": 3}


#tips = sns.load_dataset("tips")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.title("(c) Chang vs Standard SNR",weight='bold')
ax=sns.barplot(x="type", y="SNR", data=tips, capsize=0 ,palette="pastel",errorbar="se",ci=None)
#sns.swarmplot(x="type", y="SNR", data=tips, color="0", alpha=.5)
for i in range(len(Data_merged)):
    plt.plot(["chang_RT", "normal_RT"], Data_merged.iloc[i], color="black", **circle_marker,linewidth=0.5)

for i in range(len(Data_merged2)):
    plt.plot(["chang_Cryo", "normal_Cryo"], Data_merged2.iloc[i], color="black", **circle_marker,linewidth=0.5)

ax.spines['top'].set_linewidth(0)  # Top border
ax.spines['right'].set_linewidth(0)  # Right border
ax.spines['bottom'].set_linewidth(0.5)  # Bottom border
ax.spines['left'].set_linewidth(0.5)  # Left border
# Set the font to Times New Roman and font size to 8 points
#ax.set_ylim(bottom=0, top=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20,fontsize=8)
# Rest of your code...
ax.set_xlabel('')
ax.tick_params(axis='both', which='both', width=0.5,color='gray',length=2)
plt.show()


#%% SNR Chang vs Standard plot for RT and Cryp Version 2

p_address = r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\94_m_As\calculated_features\caculated_features_anatomical.csv"
p_address2= r"Z:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\94c_m_As\calculated_features\caculated_features_anatomical.csv"
#p_save = r"\\10.209.5.114\Publications\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\ChangVSStandard"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = 1/2.54  # centimeters in inches
# Set the color palette to 'Set2'

plt.figure(figsize=(18*cm/3,7.30*cm),dpi=300)
# Load the CSV data into a pandas DataFrame
data = pd.read_csv(p_address)
data2 = pd.read_csv(p_address2)



chang = data["SNR Chang"].to_frame().rename(columns={"SNR Chang": "SNR"})
normal = data["SNR Normal"].to_frame().rename(columns={"SNR Normal": "SNR"})
chang["type"] = "chang_rt"
normal["type"] = "normal_rt"



chang2 = data2["SNR Chang"].to_frame().rename(columns={"SNR Chang": "SNR"})
normal2 = data2["SNR Normal"].to_frame().rename(columns={"SNR Normal": "SNR"})
chang2["type"] = "chang_cryo"
normal2["type"] = "normal_cryo"




SNR = pd.concat([chang, normal,chang2,normal2])
SNR["SNR"] = SNR["SNR"].apply(lambda x: np.power(10,x/20))


SNR = SNR.sort_values('type')




#Data_of_selected_feature = Data_of_selected_feature.sort_values("Dataset",ascending=False)
# creating boxplots
#plt.figure(figsize=(6*cm,6*cm*np.e),dpi=300)
sns.set_style('ticks')
sns.set(font='Times New Roman', font_scale=0.9,style=None)  # Set font to Times New Roman and font size to 9
palette = 'Set2'
ax = sns.violinplot(x="type", y="SNR", data=SNR, hue="type", dodge=False,
                    palette=palette,
                    scale="width", inner=None,linewidth=1)
patches = ax.patches
#legend_colors = [patch.get_facecolor() for patch in patches[:]]

xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

sns.boxplot(x="type", y="SNR", data=SNR, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=1)
old_len_collections = len(ax.collections)
sns.stripplot(x="type", y="SNR", data=SNR, size=1.1, hue="type", palette=palette, dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.legend_.remove()
ax
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20,fontsize=8)
#ax.set_ylim(bottom=15, top=40)

#%% Plot all chang vs normal

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np


cm = 1/2.54  # centimeters in inches
# Specify the path to your Excel file
excel_file_path = r"Y:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\Chang&normal_all.xlsx"
plt.figure(figsize=(8*cm,8*cm),dpi=300)
# Read the data into a pandas DataFrame
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Drop rows with NaN values in the specified columns
#df = df.dropna(subset=['SNR-Chang (dB)', 'SNR-Standard (dB)'])

df['SNR-Chang'] = df['SNR-Chang'].apply(lambda x: np.power(10,x/20))
df['SNR-Standard'] = df['SNR-Standard'].apply(lambda x: np.power(10,x/20))

df = df.sort_values('sort')


filtered_df = df[df['Dataset'].isin(['94_m_We', '94_m_Va'])]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.title("All anatomical data",weight='bold')

# Calculate the correlation and p-value between 'SNR-Chang' and 'SNR-Standard'
#correlation, p_value = stats.pearsonr(df['SNR-Chang (dB)'], df['SNR-Standard (dB)'])
correlation, p_value = stats.spearmanr(df['SNR-Chang'], df['SNR-Standard'], nan_policy='omit',alternative='two-sided')

# Set seaborn style
sns.set_style('whitegrid')

# Create a scatter plot
#ax=sns.scatterplot(data=df, x='SNR-Chang', y='SNR-Standard',palette="gray_r",s=7)
ax=sns.scatterplot(data=df, x='SNR-Chang', y='SNR-Standard',hue="Dataset",palette="Spectral_r",s=7)




# Set title and labels including the correlation and p-value
#plt.title(f'Correlation: {correlation:.8f}, P-value: {p_value:.8f}')
plt.xlabel('SNR-Chang')
plt.ylabel('SNR-Standard')

ax.set_xlim(0,200)
ax.spines['top'].set_linewidth(0)  # Top border
ax.spines['right'].set_linewidth(0)  # Right border
ax.spines['bottom'].set_linewidth(0.5)  # Bottom border
ax.spines['left'].set_linewidth(0.5)  # Left border
# Show the plot
# Move the legend outside the plot to the right side
legend =plt.legend(title="Dataset", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5,handlelength=0.5)
legend.get_title().set_fontfamily('Times New Roman')
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')

#ax.legend_.remove()
plt.show()

#%% Plot all chang vs normal and corrolate 0 to 50 and 50 to end

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np


cm = 1/2.54  # centimeters in inches
# Specify the path to your Excel file
excel_file_path = r"Y:\2023_Kalantari_AIDAqc\outputs\QC_Final\validation\Chang&normal_all.xlsx"
plt.figure(figsize=(8*cm,8*cm),dpi=300)
# Read the data into a pandas DataFrame
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Drop rows with NaN values in the specified columns
#df = df.dropna(subset=['SNR-Chang (dB)', 'SNR-Standard (dB)'])

df['SNR-Chang'] = df['SNR-Chang'].apply(lambda x: np.power(10,x/20))
df['SNR-Standard'] = df['SNR-Standard'].apply(lambda x: np.power(10,x/20))

df = df.sort_values('sort')


filtered_df = df[df['Dataset'].isin(['94_m_We', '94_m_Va'])]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.title("All anatomical data",weight='bold')

# Create two dataframes, one for values smaller than 50 and another for values 50 or greater
df_smaller_than_50 = df[df['SNR-Chang'] < 60 & df['SNR-Standard'] < 75]

# Calculate the correlation and p-value for the group where 'SNR-Chang' < 50
correlation_smaller_than_50, p_value_smaller_than_50 = stats.spearmanr(df_smaller_than_50['SNR-Chang'], df_smaller_than_50['SNR-Standard'], nan_policy='omit', alternative='two-sided')
print("Correlation:", correlation_smaller_than_50)
print("P-value:", p_value_smaller_than_50)
