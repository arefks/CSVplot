import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# Load the NIfTI file
nifti_file = nib.load(r"C:\Users\aswen\Desktop\Code\2023_Kalantari_AIDAqc\outputs\files_4figs\GhostDTI.nii.gz")

ff=8
# Extract the data for the first direction (direction 0)
direction_0_data = nifti_file.get_fdata()[:, :, :, 0]

# Extract the data for the second direction (direction 14)
direction_1_data = nifti_file.get_fdata()[:, :, :, 14]

# Calculate the dimensions based on one of the images
cm = 1 / 2.54  # centimeters in inches
# Create a 1x2 grid of subplots with the desired size and dpi
fig, axes = plt.subplots(2, 1, figsize=(6 *cm, 7*cm), dpi=300)

# Plot the data for direction 0 in the first subplot, rotating 90 degrees clockwise
axes[0].imshow(np.rot90(direction_0_data[:, :, 5],k=1), cmap='gray', origin='lower')
axes[0].set_title('Diffusion Direction 0', fontsize=ff, fontname='Times New Roman')
axes[0].set_xticks([])  # Remove x ticks
axes[0].set_yticks([])  # Remove y ticks

# Plot the data for direction 14 in the second subplot, rotating 90 degrees clockwise
axes[1].imshow(np.rot90(direction_1_data[:, :, 5],k=1), cmap='gray', origin='lower')
axes[1].set_title('Diffusion Direction 14', fontsize=ff, fontname='Times New Roman')
axes[1].set_xticks([])  # Remove x ticks
axes[1].set_yticks([])  # Remove y ticks

# Display the subplots
plt.tight_layout()

# Save the figure to a file (optional)
#plt.savefig('diffusion_directions.png', dpi=300)

plt.show()
