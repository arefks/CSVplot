import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Create a figure with a 2x2 grid
cm = 1/2.54  # centimeters in inches
fig, axes = plt.subplots(2, 2, figsize=(9*cm, 7*cm), dpi=300)

# Define the width of the images
image_width = 0.39

# Create the axes for the "best" tSNR NIfTI image
ax0 = axes[1, 0]
im0 = ax0.imshow(tsnr_best_data[:, :, tsnr_best_data.shape[2] // 2], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
ax0.set_title("tSNR-map (best)", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax0.set_xticks([])
ax0.set_yticks([])

# Create the axes for the "worst" tSNR NIfTI image
ax1 = axes[1, 1]
im1 = ax1.imshow(tsnr_worst_data[:, :, tsnr_worst_data.shape[2] // 2], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
ax1.set_title("tSNR-map (worst)", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax1.set_xticks([])
ax1.set_yticks([])

# Create the axes for the "best" tSNR PNG image
ax2 = axes[0, 0]
ax2.imshow(tsnr_best_png)
ax2.set_title("rsfMR image", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax2.set_xticks([])
ax2.set_yticks([])

# Create the axes for the "worst" tSNR PNG image
ax3 = axes[0, 1]
ax3.imshow(tsnr_worst_png)
ax3.set_title("rsfMR image", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax3.set_xticks([])
ax3.set_yticks([])

# Add a single colorbar for the NIfTI images with more ticks
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax=cax)  # Adjust the number of ticks (11 in this case)
cbar.ax.tick_params(labelsize=8)

# Adjust the layout for better visibility
plt.tight_layout()

# Display the figure
plt.show()
