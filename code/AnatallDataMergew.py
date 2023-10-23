import os
import glob
import pandas as pd

# Step 1: Define the starting path and file pattern
start_path =  r"C:\Users\arefk\OneDrive\Desktop\Projects\Validation"
file_pattern = '*anat*.csv'

# Step 2: Find all matching CSV files in the specified directory and its subdirectories
csv_files = glob.glob(os.path.join(start_path, '**', file_pattern), recursive=True)

# Step 3: Initialize an empty DataFrame to store the extracted data
combined_df = pd.DataFrame()
combined_df1 = pd.DataFrame()
combined_df2 = pd.DataFrame()


# Step 4: Loop through the CSV files and extract the specified columns
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        selected_columns = ["FileAddress", "corresponding_img","Goasting", "SNR Chang", "SNR Normal"]
        df = df[selected_columns]
        
        # Concatenate the current DataFrame with the combined DataFrame
        combined_df1 = pd.concat([combined_df1, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        selected_columns = ["FileAddress", "img name","Goasting", "SNR Chang", "SNR Normal"]
        df = df[selected_columns]
        
        # Concatenate the current DataFrame with the combined DataFrame
        combined_df2 = pd.concat([combined_df2, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        
combined_df2=combined_df2.rename({"img name": "corresponding_img"}, axis=1)
combined_df = pd.concat([combined_df1, combined_df2], ignore_index=True)

# Step 5: Print the combined DataFrame
print(combined_df)

# Optionally, you can save the combined DataFrame to a CSV file
p = r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs"
combined_df.to_csv(os.path.join(p,'combined_data_anatF.csv'), index=False)