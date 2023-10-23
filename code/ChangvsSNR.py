#%% Plot all chang vs normal

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt

cm = 1/2.54  # centimeters in inches
# Specify the path to your Excel file
img_path = r"C:\Users\arefk\OneDrive\Desktop\Projects\Validation\*\manual_slice_inspection"
new_path = r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs\ChangVSNormal"
excel_file_path = r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs\combined_data_anat.csv"
plt.figure(figsize=(10*cm,10*cm),dpi=300)
# Read the data into a pandas DataFrame
df = pd.read_csv(excel_file_path)

# Drop rows with NaN values in the specified columns
#df = df.dropna(subset=['SNR-Chang (dB)', 'SNR-Standard (dB)'])

df['SNR Chang'] = df['SNR Chang'].apply(lambda x: np.power(10,x/20))
df['SNR Normal'] = df['SNR Normal'].apply(lambda x: np.power(10,x/20))
#f['FileAddress2'] = df['FileAddress'].apply(lambda x:x.split(os.path.sep)[0])
df_filterd = df[df["SNR Normal"] < 19]
df_filterd= df_filterd[df_filterd["SNR Chang"]>90]
df_filterd['names'] = df_filterd['FileAddress'].apply(lambda x:x.split("mri")[1].split(os.path.sep)[1])


df_filterd.to_csv(r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs\combined_data_anat_changVSstandard_Check.csv")
P = []
for i in df_filterd["corresponding_img"]:
    print(i)
    
    old_add = glob.glob(os.path.join(img_path,i),recursive=True)
    new_add = os.path.join(new_path,i)
    shutil.copyfile(old_add[0],new_add)
    

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.title("All anatomical data",weight='bold', fontsize=10)

# Calculate the correlation and p-value between 'SNR-Chang' and 'SNR-Standard'

# Set seaborn style
sns.set_style('whitegrid')

ax=sns.scatterplot(data=df_filterd, x='SNR Chang', y='SNR Normal',hue="names",palette="Spectral_r",s=7)
ax.set_title("All anatomical data", weight='bold', fontsize=11)




# Set title and labels including the correlation and p-value
#plt.title(f'Correlation: {correlation:.8f}, P-value: {p_value:.8f}')
plt.xlabel('SNR-Chang')
plt.ylabel('SNR-Standard')

ax.set_xlim(0,200)
ax.set_ylim(0,140)

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