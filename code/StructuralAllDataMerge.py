import os
import glob
import pandas as pd

# Step 1: Define the starting path and file pattern
start_path = r"C:\Users\aswen\Documents\Data\2023_Kalantari_AIDAqc\outputs\validation\QC_Chang"
file_pattern = '*structu*.csv'

# Step 2: Find all matching CSV files in the specified directory and its subdirectories
csv_files = glob.glob(os.path.join(start_path, '**', file_pattern), recursive=True)

# Step 3: Initialize an empty DataFrame to store the extracted data
combined_df = pd.DataFrame()

# Step 4: Loop through the CSV files and extract the specified columns
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        selected_columns = ["FileAddress", "SNR Chang", "SNR Normal", "Displacement factor (std of Mutual information)", "Goasting"]
        df = df[selected_columns]
        
        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Step 5: Calculate the mean and standard deviation for SNR Chang, SNR Normal, and Displacement factor
mean_snr_chang = combined_df["SNR Chang"].mean()
std_snr_chang = combined_df["SNR Chang"].std()

mean_snr_normal = combined_df["SNR Normal"].mean()
std_snr_normal = combined_df["SNR Normal"].std()

mean_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].mean()
std_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].std()

# Print the results
print(f"Mean SNR Chang: {mean_snr_chang}")
print(f"Standard Deviation SNR Chang: {std_snr_chang}")
print(f"Mean SNR Normal: {mean_snr_normal}")
print(f"Standard Deviation SNR Normal: {std_snr_normal}")
print(f"Mean Displacement Factor: {mean_displacement_factor}")
print(f"Standard Deviation Displacement Factor: {std_displacement_factor}")

# Optionally, you can save the combined DataFrame to a CSV file
p = r"Z:\2023_Kalantari_AIDAqc\outputs\figs"
combined_df.to_csv(os.path.join(p,'combined_data_struct.csv'), index=False)