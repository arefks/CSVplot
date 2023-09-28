import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from the CSV file
df = pd.read_csv(r"Z:\2023_Kalantari_AIDAqc\outputs\figs\MUplot.csv", header=None, names=['Shift (Voxel)', 'Severe motion', 'No motion'], skiprows=1)

# Set the seaborn style and palette
#sns.set(style='whitegrid', palette='Set1')

# Set font properties
font_properties = {'family': 'Times New Roman', 'size': 8}
font_properties2 = {'family': 'Times New Roman', 'size': 6}

cm = 1/2.54  # centimeters in inches

# Create the plot
fig, ax = plt.subplots(figsize=(7.01*cm, 3.21*cm), dpi=300)
ax.plot(df['Shift (Voxel)'], df['Severe motion'], label='Severe motion', linewidth=1)  # Adjust the line width
ax.plot(df['Shift (Voxel)'], df['No motion'], label='No motion', linewidth=1, color='blue')  # Adjust the line width

# Set axis labels
ax.set_xlabel('Shift (Voxel)', **font_properties)
ax.set_ylabel('Mutual information (u.a)', **font_properties)

# Set axis ticks font
ax.tick_params(axis='both', which='both', width=0.5, color='gray', length=2)
for tick in ax.get_xticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)
for tick in ax.get_yticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(8)

# Set legend font and remove the legend border
legend = ax.legend(prop=font_properties2, frameon=False)

# Customize the border linewidth
ax.spines['top'].set_linewidth(0)     # Top border
ax.spines['right'].set_linewidth(0)   # Right border
ax.spines['bottom'].set_linewidth(0.5)  # Bottom border
ax.spines['left'].set_linewidth(0.5)   # Left border

# Show the plot
plt.show()
