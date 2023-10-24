import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Define the directory path where the NIfTI and PNG files are located
path = r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs"  # Replace with your actual directory path

# File names for NIfTI images (assuming they represent motion data)
motion_best_filename = "MotionBest.nii.gz"
motion_worst_filename = "MotionWorst.nii.gz"

# Load the NIfTI files
motion_best = nib.load(os.path.join(path, motion_best_filename))
motion_worst = nib.load(os.path.join(path, motion_worst_filename))

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

motion_worst_data = motion_worst_data[:, :, slice_index2]

# Normalize the data to the range of 0 to 1
norm = Normalize(vmin=motion_best_data.min(), vmax=motion_best_data.max())
motion_best_data = norm(motion_best_data)
motion_worst_data = norm(motion_worst_data)

# Create a figure with GridSpec
cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(9*cm, 6.68*cm), dpi=300)
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Create the axes for the "Motion Data (Best)" NIfTI image
ax0 = plt.subplot(gs[0])
im0 = ax0.imshow(motion_best_data, cmap='viridis', origin='lower', vmin=0, vmax=1)
ax0.set_title("rsfMRI low motion", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax0.set_xticks([])
ax0.set_yticks([])

# Create the axes for the "Motion Data (Worst)" NIfTI image
ax1 = plt.subplot(gs[1])
im1 = ax1.imshow(motion_worst_data, cmap='viridis', origin='lower', vmin=0, vmax=1)
ax1.set_title("rsfMRI high motion", fontsize=10, fontweight='bold', fontname='Times New Roman')
ax1.set_xticks([])
ax1.set_yticks([])

# Add a single colorbar for the NIfTI images with more ticks
divider0 = make_axes_locatable(ax1)
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cbar0 = plt.colorbar(im0, cax=cax0, ticks=np.linspace(0, 1, 6))  # Adjust the number of ticks (6 in this case)
cbar0.ax.tick_params(labelsize=8)

# Adjust the layout for better visibility
plt.tight_layout()

# Display the figure
plt.show()
