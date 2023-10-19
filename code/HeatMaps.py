import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load your Excel file into a Pandas DataFrame
excel_file = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs\Kappa_and_F1Score_results2.xlsx"
df = pd.read_excel(excel_file)

# Set the user-defined threshold value
user_threshold = 3# Replace with the user-set threshold value

# Filter the DataFrame based on the user-set threshold
filtered_df = df[df['Thresold'] == user_threshold]
cm = 1/2.54  # centimeters in inches

# Create two subplots for Kappa and F1 scores heatmaps
fig, axes = plt.subplots(2, 1, figsize=(18*cm, 10*cm), dpi=300)

# Specify the font properties
font_properties = fm.FontProperties(family='Times New Roman', size=8)
font_properties2 = fm.FontProperties(family='Times New Roman', size=10)


for i, metric in enumerate(['Kappa', 'F1_score']):
    ax = axes[i]
    pivot_df = filtered_df.pivot(index='Sequence', columns='Dataset', values=metric)
    
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, ax=ax)

    # Set font properties for labels, titles, and annotations
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties)
    ax.set_xlabel('Datasets', fontproperties=font_properties)
    ax.set_ylabel('Sequences', fontproperties=font_properties)
    ax.set_title(f'{metric} Heatmap (Threshold = {user_threshold})', fontsize=10, fontproperties=font_properties2, fontweight='bold')

    # Set the color bar legend font size and font properties
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontproperties=font_properties)
        
    for text in ax.texts:
        text.set_size(8)
        text.set_font("Times New Roman")
        
        
# Show the plots
plt.tight_layout()
plt.show()
