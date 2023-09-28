import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from the CSV file
df = pd.read_csv(r"Z:\2023_Kalantari_AIDAqc\outputs\figs\Ghostplot.csv", header=None, names=['Shift (Voxel)', 'Severe motion', 'No motion'], skiprows=1)

# Set the seaborn style and palette
sns.set(style='whitegrid', palette='Set1')

# Set font properties
font_properties = {'family': 'Times New Roman', 'size': 8}
font_properties2 = {'family': 'Times New Roman', 'size': 8}

cm = 1/2.54  # centimeters in inches

# Create the first plot (No ghost)
fig, ax1 = plt.subplots(figsize=(4.5*cm, 4.91*cm), dpi=300)
ax1.plot(df['Shift (Voxel)'], df['No motion'], label='No ghost', linewidth=1, color='red')
ax1.set_xlabel('Shift (Voxel)', **font_properties)
ax1.set_ylabel('Mutual Information (u.a)', **font_properties)
ax1.tick_params(axis='both', which='both', width=0.5, color='gray', length=2)
for tick in ax1.get_xticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)
for tick in ax1.get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)
ax1.spines['top'].set_linewidth(0)     # Top border
ax1.spines['right'].set_linewidth(0)   # Right border
ax1.spines['bottom'].set_linewidth(0.5)  # Bottom border
ax1.spines['left'].set_linewidth(0.5)   # Left border

#ax1.legend(prop=font_properties2, frameon=False)

# Create the second plot (Ghost)
fig, ax2 = plt.subplots(figsize=(4.5*cm, 4.91*cm), dpi=300)
ax2.plot(df['Shift (Voxel)'], df['Severe motion'], label='Ghost', linewidth=1, color='blue')
ax2.set_xlabel('Shift (Voxel)', **font_properties)
ax2.set_ylabel('Mutual Information (u.a)', **font_properties)
ax2.tick_params(axis='both', which='both', width=0.5, color='gray', length=2)
for tick in ax2.get_xticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)
for tick in ax2.get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)
ax2.spines['top'].set_linewidth(0)     # Top border
ax2.spines['right'].set_linewidth(0)   # Right border
ax2.spines['bottom'].set_linewidth(0.5)  # Bottom border
ax2.spines['left'].set_linewidth(0.5)   # Left border

#ax2.legend(prop=font_properties2, frameon=False)

# Show the plots
plt.show()
