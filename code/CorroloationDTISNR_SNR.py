import pandas as pd
import os

# Step 1: Read three CSV files
path = r"Z:\2023_Kalantari_AIDAqc\outputs\figs"  # Use a raw string (r) to avoid escape characters
func_df = pd.read_csv(os.path.join(path, 'combined_data_func.csv'))
struct_df = pd.read_csv(os.path.join(path, 'combined_data_struct.csv'))
anat_df = pd.read_csv(os.path.join(path, 'combined_data_anat.csv'))

# Step 4: Sort all dataframes based on the FileAddress column
func_df.sort_values(by='FileAddress', inplace=True)
struct_df.sort_values(by='FileAddress', inplace=True)
anat_df.sort_values(by='FileAddress', inplace=True)

# Step 5: Process the FileAddress column
def process_file_address(file_address):
    elements = file_address.split('\\')  # Use '\\' to split on backslash
    return '\\'.join(elements[:-1])  # Use '\\' to join elements with backslash

struct_df['FileAddress'] = struct_df['FileAddress'].apply(process_file_address)
anat_df['FileAddress'] = anat_df['FileAddress'].apply(process_file_address)

# Step 6: Create a new dataframe
common_file_addresses = set(anat_df['FileAddress']).intersection(set(struct_df['FileAddress']))
result_data = []

for file_address in common_file_addresses:
    anat_rows = anat_df[anat_df['FileAddress'] == file_address]
    struct_rows = struct_df[struct_df['FileAddress'] == file_address]
    
    # Calculate the average of 'SNR Chang' and 'tSNR (Averaged Brain ROI)' values, ignoring NaNs
    avg_snr_chang = anat_rows['SNR Normal'].mean()
    avg_tsnr_avg_brain_roi = struct_rows['SNR Normal'].mean()
    
    result_data.append({
        'Common FileAddress': file_address,
        'Average SNR Chang': avg_snr_chang,
        'Average tSNR (Averaged Brain ROI)': avg_tsnr_avg_brain_roi
    })

# Create the result DataFrame
result_df = pd.DataFrame(result_data)

# Print the result
print(result_df)
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# Calculate the Spearman correlation coefficient and p-value
corr, p_value = spearmanr(result_df['Average SNR Chang'], result_df['Average tSNR (Averaged Brain ROI)'])

# Define the centimeters to inches conversion
cm = 1/2.54  # centimeters in inches
sns.set_style("ticks")
# Set the font style to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create the plot

fig, ax = plt.subplots(figsize=(4.5*cm, 5*cm), dpi=300)

# Use Seaborn's scatterplot function
sns.scatterplot(x='Average SNR Chang', y='Average tSNR (Averaged Brain ROI)',
                data=result_df, palette='Set2', s=10)

plt.xlabel('Anatomical SNR standard (dB)', fontsize=8)
plt.ylabel('Structural SNR standard (dB)', fontsize=8)

# Set font size for tick labels
ax.tick_params(axis='both', which='both', labelsize=8)

# Remove right and upper borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set border widths for the remaining borders
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)

# Add horizontal lines from yticks
ax.yaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=0.5)
ax.yaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)

# Add horizontal lines from yticks
ax.xaxis.grid(True, linestyle='-', which='major', color='gray', linewidth=0.5)
ax.xaxis.grid(True, linestyle='--', which='minor', color='gray', linewidth=0.5)
ax.tick_params(axis='both', which='both', width=0.5,color='gray',length=2)
ax.set_title("(d) SNR struct vs anat",weight='bold',fontsize=10)
plt.show()
