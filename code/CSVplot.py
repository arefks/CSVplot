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

# Read the CSV files and store them in dictionaries
anatomical_data = read_csv_files(anatomical_files)
structural_data = read_csv_files(structural_files)
functional_data = read_csv_files(functional_files)

#%% Read files

features_to_compare = ["SNR Chang", "SNR Normal", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)"]

temp = pd.DataFrame()
DataSNR = pd.DataFrame()
temp_data = pd.DataFrame()
dd = 0
for key in anatomical_data:
    temp_data["SNR Normal"] = anatomical_data[key]["anatomical"]["SNR Normal"]
    temp_data["Field Strength"] = anatomical_files[dd].split(os.sep)[-3]
    dd = dd +1
    DataSNR = pd.concat([DataSNR, temp_data], ignore_index=True)


#%%
plt.figure(figsize=(30, 10))
sns.set_style('white')
palette = 'Set2'
ax = sns.violinplot(x="Field Strength", y="SNR Normal", data=DataSNR, hue="Field Strength", dodge=False,
                    palette=palette,
                    scale="width", inner=None)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

sns.boxplot(x="Field Strength", y="SNR Normal", data=DataSNR, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x="Field Strength", y="SNR Normal", data=DataSNR, hue="Field Strength", palette=palette, dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.legend_.remove()
plt.savefig(os.path.join(Path,"FIGX.svg"), format='svg', bbox_inches='tight')
plt.show()    
#%%

# Example data (replace with your data)
# ... (assuming DataSNR and groups are already defined)
groups = DataSNR["Field Strength"].unique()
# Calculate pairwise p-values using t-test
p_values = np.zeros((len(groups), len(groups)))

for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        snr_group1 = DataSNR[DataSNR["Field Strength"] == group1]["SNR Normal"]
        snr_group2 = DataSNR[DataSNR["Field Strength"] == group2]["SNR Normal"]
        t_stat, p_value = ttest_ind(snr_group1, snr_group2)
        p_values[i, j] = p_value

# Apply Bonferroni correction
corrected_p_values = multipletests(p_values.flatten(), method='bonferroni')[1]
corrected_p_matrix = corrected_p_values.reshape(len(groups), len(groups))

# Mask out the diagonal and upper triangle for better visualization
mask = np.triu(np.ones_like(corrected_p_matrix), k=1)

plt.figure(figsize=(12, 8))
sns.heatmap(corrected_p_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=groups, yticklabels=groups, mask=mask)

plt.title("Pairwise Comparisons Heatmap with Bonferroni Correction")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(Path,"HEATMAP_pairwiseComparison.svg"), format='svg', bbox_inches='tight')
plt.show()

#%%

# Example data (replace with your data)
# ... (assuming DataSNR and groups are already defined)

# Calculate pairwise p-values using t-test
p_values = np.zeros((len(groups), len(groups)))

for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        snr_group1 = DataSNR[DataSNR["Field Strength"] == group1]["SNR Normal"]
        snr_group2 = DataSNR[DataSNR["Field Strength"] == group2]["SNR Normal"]
        t_stat, p_value = ttest_ind(snr_group1, snr_group2)
        p_values[i, j] = p_value

# Apply Bonferroni correction
corrected_p_values = multipletests(p_values.flatten(), method='bonferroni')[1]
corrected_p_matrix = corrected_p_values.reshape(len(groups), len(groups))

# Create a DataFrame to display all comparisons and p-values
comparison_pvalues = []

for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:
            comparison_pvalues.append((group1, group2, corrected_p_matrix[i, j]))

comparison_pvalues_df = pd.DataFrame(comparison_pvalues, columns=["Group 1", "Group 2", "Corrected P-Value"])

# Save the DataFrame to a beautiful table (you can choose the format you prefer)
# For example, save to a CSV file
comparison_pvalues_df.to_csv(os.path.join(Path,"significant_pairs.csv"), index=False)

# You can also display the DataFrame if you're working in an interactive environment
print(comparison_pvalues_df)

    
    
    
    
    
    
    