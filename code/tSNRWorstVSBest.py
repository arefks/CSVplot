import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Define the directory path where the NIfTI and PNG files are located
path = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs"  # Replace with your actual directory path

# File names for NIfTI images
tsnr_best_filename = "TSNRmap_best.nii.gz"
tsnr_worst_filename = "TSNRmap_worst.nii.gz"

# Load the NIfTI files
tsnr_best = nib.load(os.path.join(path, tsnr_best_filename))
tsnr_worst = nib.load(os.path.join(path, tsnr_worst_filename))

# Get the NIfTI data as NumPy arrays
tsnr_best_data = tsnr_best.get_fdata()
tsnr_worst_data = tsnr_worst.get_fdata()

# Rotate the data arrays 90 degrees counterclockwise
tsnr_best_data = np.rot90(tsnr_best_data, k=-1)
tsnr_worst_data = np.roll(tsnr_worst_data, shift=5, axis=1)
tsnr_worst_data = np.rot90(tsnr_worst_data, k=1)

# Determine a common color scale range for both rotated images
vmin = min(tsnr_best_data.min(), tsnr_worst_data.min())
vmax = max(tsnr_best_data.max(), tsnr_worst_data.max())

# Load the PNG images
tsnr_best_png = plt.imread(os.path.join(path, "TSNRbest.png"))
tsnr_worst_png = plt.imread(os.path.join(path, "TSNRworst.png"))
ff=8
# Create a figure with a 2x2 grid
cm = 1/2.54  # centimeters in inches

# Calculate the dimensions based on one of the images
image_width = 1.7  # Adjust this value as needed
image_height = (image_width * tsnr_best_data.shape[0]) / tsnr_best_data.shape[1]
width = 9*cm
height = 9*cm


fig= plt.figure( figsize=(width , height), dpi=300)
gs = GridSpec(2, 2, width_ratios=[1.00, 1.01])

# Create the axes for the "Motion Data (Best)" NIfTI image
ax0 = plt.subplot(gs[1])
# Create the axes for the "best" tSNR NIfTI image
im0 = ax0.imshow(tsnr_best_data[:, :, tsnr_best_data.shape[2] // 2], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
ax0.set_title("tSNR-map (best)", fontsize=ff, fontname='Times New Roman')
ax0.set_xticks([])
ax0.set_yticks([])

# Create the axes for the "worst" tSNR NIfTI image
ax1 = plt.subplot(gs[3])
im1 = ax1.imshow(tsnr_worst_data[:, :, tsnr_worst_data.shape[2] // 2], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
ax1.set_title("tSNR-map (worst)", fontsize=ff, fontname='Times New Roman')
ax1.set_xticks([])
ax1.set_yticks([])

# Create the axes for the "best" tSNR PNG image
ax2 = plt.subplot(gs[0])
ax2.imshow(tsnr_best_png)
ax2.set_title("rsfMR image (best)", fontsize=ff, fontname='Times New Roman')
ax2.set_xticks([])
ax2.set_yticks([])

# Create the axes for the "worst" tSNR PNG image
ax3 = plt.subplot(gs[2])
ax3.imshow(tsnr_worst_png)
ax3.set_title("rsfMR image (worst)", fontsize=ff, fontname='Times New Roman')
ax3.set_xticks([])
ax3.set_yticks([])

# Add a single colorbar for the NIfTI images with more ticks
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax=cax)  # Adjust the number of ticks (11 in this case)

# Set the colorbar tick labels to Times New Roman with a font size of 8 points
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=8, fontname='Times New Roman')



# Add a single colorbar for the NIfTI images with more ticks
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax=cax)  # Adjust the number of ticks (11 in this case)

# Set the colorbar tick labels to Times New Roman with a font size of 8 points
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=8, fontname='Times New Roman')

# Adjust the layout for better visibility
plt.tight_layout(h_pad=-3)

# Display the figure
plt.show()
