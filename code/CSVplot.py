import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import glob

# Function to read CSV files and concatenate them into a single DataFrame
def read_csv_files(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs)

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

# List of CSV files for each data type
anatomical_files = glob.glob("*caculated_features_anatomical.csv")
structural_files = glob.glob("*caculated_features_structural.csv")
functional_files = glob.glob("*caculated_features_functional.csv")

# Read and concatenate the CSV files for each data type
anatomical_data = read_csv_files(anatomical_files)
structural_data = read_csv_files(structural_files)
functional_data = read_csv_files(functional_files)

# Remove NaN values
anatomical_data = anatomical_data.dropna()
structural_data = structural_data.dropna()
functional_data = functional_data.dropna()

# Perform t-tests and ANOVAs for different features
features_to_compare = ["SNR Chang", "SNR Normal", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)"]

for feature in features_to_compare:
    print(f"\nFeature: {feature}\n")
    
    # t-test between different field strengths (assuming the field strength information is in the 'sequence name' column)
    field_strengths = anatomical_data["sequence name"].unique()
    for strength in field_strengths:
        data_group = anatomical_data[anatomical_data["sequence name"] == strength][feature]
        print(f"t-test results for {strength} vs. other field strengths:")
        for other_strength in field_strengths[field_strengths != strength]:
            other_data_group = anatomical_data[anatomical_data["sequence name"] == other_strength][feature]
            t_stat, p_value = ttest_ind(data_group, other_data_group)
            print(f"{strength} vs. {other_strength}: p-value = {p_value:.4f}")

    # ANOVA test between all field strengths
    f_stat, p_value = f_oneway(*[anatomical_data[anatomical_data["sequence name"] == strength][feature] for strength in field_strengths])
    print(f"\nANOVA results for all field strengths: p-value = {p_value:.4f}")

    # Plot the comparison for the current feature
    compare_and_plot(anatomical_data, feature, "sequence name")

# You can repeat the same for structural and functional data using the corresponding dataframes.

# If you want to combine all data for further analysis, you can use:
combined_data = pd.concat([anatomical_data, structural_data, functional_data])

# Save the combined data to a new CSV file
combined_data.to_csv("combined_data.csv", index=False)
