import os
import glob
import pandas as pd

# Step 1: Define the starting path and file pattern
start_path = r"C:\Users\aswen\Desktop\Code\Validation2"
file_pattern = '*func*.csv'

# Step 2: Find all matching CSV files in the specified directory and its subdirectories
csv_files = glob.glob(os.path.join(start_path, '**', file_pattern), recursive=True)

# Step 3: Initialize an empty DataFrame to store the extracted data
combined_df = pd.DataFrame()

# Step 4: Loop through the CSV files and extract the specified columns
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        selected_columns = ["FileAddress", "tSNR (Averaged Brain ROI)", "Displacement factor (std of Mutual information)", "Goasting"]
        df = df[selected_columns]
        
        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Step 5: Calculate the mean, standard deviation, minimum, and maximum for tSNR and Displacement factor
mean_tsnr = combined_df["tSNR (Averaged Brain ROI)"].mean()
std_tsnr = combined_df["tSNR (Averaged Brain ROI)"].std()
min_tsnr = combined_df["tSNR (Averaged Brain ROI)"].min()
max_tsnr = combined_df["tSNR (Averaged Brain ROI)"].max()

mean_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].mean()
std_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].std()
min_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].min()
max_displacement_factor = combined_df["Displacement factor (std of Mutual information)"].max()

# Print the results
print(f"Mean tSNR: {mean_tsnr}")
print(f"Standard Deviation tSNR: {std_tsnr}")
print(f"Minimum tSNR: {min_tsnr}")
print(f"Maximum tSNR: {max_tsnr}")

print(f"Mean Displacement Factor: {mean_displacement_factor}")
print(f"Standard Deviation Displacement Factor: {std_displacement_factor}")
print(f"Minimum Displacement Factor: {min_displacement_factor}")
print(f"Maximum Displacement Factor: {max_displacement_factor}")

# Optionally, you can save the combined DataFrame to a CSV file
p = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs"
combined_df.to_csv(os.path.join(p,'combined_data_func.csv'), index=False)
