import glob
import os
import pandas as pd

# Step 1: Accept an initial path as input and look for "voting.csv" files
initial_path = r"C:\Users\arefk\OneDrive\Desktop\Projects\validation\94_r_Fr"  # Replace with the actual path
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
combined_df = pd.concat(dataframes, ignore_index=True)

# Step 3: Filter rows based on the "majority vote" column
threshold = 3
VoteTemp = combined_df[combined_df["Voting outliers (from 5)"] >= threshold]

# Step 4: List all PNG images in the "manual_slice_inspection" folder
manual_slice_inspection_path = os.path.join(initial_path, "manual_slice_inspection")
manual_slice_images = [file for file in os.listdir(manual_slice_inspection_path) if file.endswith(".png")]

# Create a DataFrame for "manual_slice_inspection"
data = {"manual_slice_inspection": manual_slice_images}
ManualRaterTemp = pd.DataFrame(data)

# Step 5: Create columns for each folder in "validation_folders"
for folder in validation_folders:
    folder_name = os.path.basename(folder)
    folder_images = [file for file in os.listdir(folder) if file.endswith(".png")]
    
    # Create a column with 1 if the image is in the folder, 0 otherwise
    ManualRaterTemp[folder_name] = ManualRaterTemp["manual_slice_inspection"].apply(lambda x: 1 if x in folder_images else 0)

# Now you have "VoteTemp" and "ManualRaterTemp" dataframes as described in steps 3 and 5 with the updated requirements.
