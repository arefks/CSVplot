import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import glob

# Function to read CSV files and store them in a dictionary


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

# List of CSV files for each data type
anatomical_files = glob.glob("**/*caculated_features_anatomical.csv", recursive=True)
structural_files = glob.glob("**/*caculated_features_structural.csv", recursive=True)
functional_files = glob.glob("**/*caculated_features_functional.csv", recursive=True)

# Read the CSV files and store them in dictionaries
anatomical_data = read_csv_files(anatomical_files)
structural_data = read_csv_files(structural_files)
functional_data = read_csv_files(functional_files)

# Perform t-tests and ANOVAs for different features
features_to_compare = ["SNR Chang", "SNR Normal", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)"]

for feature in features_to_compare:
    print(f"\nFeature: {feature}\n")
    
    # t-test between different field strengths (assuming the field strength information is in the 'sequence name' column)
    field_strengths = set()
    for subject_data in anatomical_data.values():
        if "anatomical" in subject_data:
            field_strengths.update(subject_data["anatomical"]["sequence name"].unique())

    for strength in field_strengths:
        data_group = [subject_data["anatomical"][feature] for subject_data in anatomical_data.values() if "anatomical" in subject_data and (strength in subject_data["anatomical"]["sequence name"].values)]
        print(f"t-test results for {strength} vs. other field strengths:")
        for other_strength in field_strengths - {strength}:
            other_data_group = [subject_data["anatomical"][feature] for subject_data in anatomical_data.values() if "anatomical" in subject_data and (other_strength in subject_data["anatomical"]["sequence name"].values)]
            t_stat, p_value = ttest_ind(data_group, other_data_group)
            print(f"{strength} vs. {other_strength}: p-value = {p_value:.4f}")

    # ANOVA test between all field strengths
    all_data = [data for subject_data in anatomical_data.values() if "anatomical" in subject_data for data in subject_data["anatomical"][feature]]
    group_labels = [strength for strength in field_strengths for _ in range(len(anatomical_data.values())) if "anatomical" in subject_data and (strength in subject_data["anatomical"]["sequence name"].values)]
    f_stat, p_value = f_oneway(*all_data)
    print(f"\nANOVA results for all field strengths: p-value = {p_value:.4f}")

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data)

    # Plot the comparison for the current feature
    compare_and_plot(combined_data, feature, "sequence name")

# You can repeat the same for structural and functional data using the corresponding dataframes.

# If you want to combine all data for further analysis, you can use:
all_data = pd.concat([data for subject_data in anatomical_data.values() if "anatomical" in subject_data for data in subject_data["anatomical"].values()])
# Save the combined data to a new CSV file
all_data.to_csv("combined_data.csv", index=False)
