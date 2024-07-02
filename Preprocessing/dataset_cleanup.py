# Importing libraries
import os
import cv2
import numpy as np
from tqdm import tqdm

# Parameters
RGB_PATCHES_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/RGB images"
DEM_PATCHES_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/DEM images"
DSM_PATCHES_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/DSM images"
MASK_PATCHES_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/Masks"

# Function to delete corresponding patches
def delete_corresponding_patches(patch_filename):
    # Construct full paths for the corresponding patches
    rgb_patch_path = os.path.join(RGB_PATCHES_DIR, patch_filename)
    dem_patch_path = os.path.join(DEM_PATCHES_DIR, patch_filename)
    dsm_patch_path = os.path.join(DSM_PATCHES_DIR, patch_filename)
    mask_patch_path = os.path.join(MASK_PATCHES_DIR, patch_filename)
    
    # Delete the files if they exist
    for path in [rgb_patch_path, dem_patch_path, dsm_patch_path, mask_patch_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted: {path}")

# Iterate through the mask patches
for mask_patch_filename in tqdm(os.listdir(MASK_PATCHES_DIR), desc="Processing mask patches"):
    mask_patch_path = os.path.join(MASK_PATCHES_DIR, mask_patch_filename)
    
    # Load the mask patch
    mask_patch = cv2.imread(mask_patch_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the mask patch is completely black
    if mask_patch is not None and np.all(mask_patch == 0):
        # Delete the mask and corresponding patches
        delete_corresponding_patches(mask_patch_filename)

print("Processing completed.")
