import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os

# Load your Excel file into a Pandas DataFrame
excel_file = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs\Kappa_and_F1Score_results.xlsx"
#excel_file = r"C:\Users\arefk\OneDrive\Desktop\Projects\2023_Kalantari_AIDAqc\outputs\files_4figs\Kappa_and_F1Score_resultsTemp.xlsx"
Save = "yes"

df = pd.read_excel(excel_file)

# Set the user-defined threshold value
user_threshold = 1  # Replace with the user-set threshold value

# Filter the DataFrame based on the user-set threshold
filtered_df = df[df['Thresold'] == user_threshold]
cm = 1/2.54  # centimeters in inches

# Create two subplots for Kappa and F1 scores heatmaps
fig, axes = plt.subplots(2, 1, figsize=(20*cm, 10*cm), dpi=300)
#sns.set_style('whitegrid')
# Specify the font properties
font_properties = fm.FontProperties(family='Times New Roman', size=8)
font_properties2 = fm.FontProperties(family='Times New Roman', size=10)
Title = ["(a) Fleiss kappa score: inter rater reliability "," (b) F1_score: raters vs AIDAqc"]
for i, metric in enumerate(['Kappa', 'F1_score']):
    ax = axes[i]
    pivot_df = filtered_df.pivot(index='Sequence', columns='Dataset', values=metric)
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df['std'] = pivot_df.std(axis=1)
    t=Title[i]
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, ax=ax)

    # Set font properties for labels, titles, and annotations
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties)
    ax.set_xlabel('Datasets', fontproperties=font_properties)
    ax.set_ylabel('Sequences', fontproperties=font_properties)
    ax.set_title(f'{t} ', fontsize=10, fontproperties=font_properties2, fontweight='bold')
    ax.set(xlabel=None)
    # Set the color bar legend font size and font properties
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontproperties=font_properties)
    
    # Customize the color bar ticks (increase the number of ticks)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    a = []
    for text in ax.texts:
        a.append(float(text.get_text()))
        
    

    for text in ax.texts:
        text.set_font("Times New Roman")
        text.set_size(8)
        

# Show the plots
plt.tight_layout()

if Save == "yes":
    plt.savefig(os.path.join(os.path.dirname(excel_file),"SVG_HEATMAP.svg"))        
plt.show()
