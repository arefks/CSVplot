import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Define the directory path where the NIfTI files are located
path = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs"  # Replace with your actual directory path

# File names for NIfTI images (assuming they represent motion data)
motion_best_filename = "MotionBest.nii.gz"
motion_worst_filename = "MotionWorst2.nii.gz"

# Load the NIfTI files
motion_worst = nib.load(os.path.join(path, motion_worst_filename))
motion_best = nib.load(os.path.join(path, motion_best_filename))

# Get the NIfTI data as NumPy arrays
motion_best_data = motion_best.get_fdata()
motion_worst_data = motion_worst.get_fdata()

# Rotate the data arrays 90 degrees counterclockwise
motion_best_data = np.rot90(motion_best_data, k=-1)
motion_worst_data = np.rot90(motion_worst_data, k=1)

# Average over the 4th dimension and select the middle slice
motion_best_data = np.mean(motion_best_data, axis=3)
motion_worst_data = np.mean(motion_worst_data, axis=3)
slice_index1 = motion_best_data.shape[2] // 2
slice_index2 = motion_worst_data.shape[2] // 2

motion_best_data = motion_best_data[:, :, slice_index1]
motion_best_data = np.rot90(np.rot90(motion_best_data, k=1), k=1)

motion_worst_data = motion_worst_data[:, :, slice_index2 + 4]

# Calculate the dimensions based on one of the images
cm = 1 / 2.54  # centimeters in inches
total_width = 6  # Total width in cm
aspect_ratio = motion_best_data.shape[1] / motion_best_data.shape[0]
image_height = total_width / 2 / aspect_ratio  # Divide by 2 for two images side by side

# Normalize the data to the range of 0 to 1
norm1 = Normalize(vmin=motion_best_data.min(), vmax=motion_best_data.max())
norm2 = Normalize(vmin=motion_worst_data.min(), vmax=motion_worst_data.max())

motion_best_data = norm1(motion_best_data)
motion_worst_data = norm2(motion_worst_data)

# Create a figure with GridSpec
fig = plt.figure(figsize=(6 * cm, 7 * cm), dpi=300)
gs = GridSpec(2, 1, height_ratios =[1.5,1])

# Create the axes for the "Motion Data (Best)" NIfTI image
ax0 = plt.subplot(gs[0])
im0 = ax0.imshow(motion_best_data, cmap='viridis', origin='lower', vmin=0, vmax=1)
ax0.set_title("rsfMRI low motion", fontsize=8, fontname='Times New Roman')
ax0.set_xticks([])
ax0.set_yticks([])

# Create the axes for the "Motion Data (Worst)" NIfTI image
ax1 = plt.subplot(gs[1])
im1 = ax1.imshow(motion_worst_data, cmap='viridis', origin='lower', vmin=0, vmax=1)
ax1.set_title("rsfMRI high motion", fontsize=8, fontname='Times New Roman')
ax1.set_xticks([])
ax1.set_yticks([])

# Add a single colorbar for the NIfTI images with more ticks
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax=cax)  # Adjust the number of ticks (11 in this case)

# Set the colorbar tick labels to Times New Roman with a font size of 8 points
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=8, fontname='Times New Roman')

# Adjust the layout for better visibility
plt.tight_layout()

# Display the figure
plt.show()
