# Importing libraries
import os
import cv2
from tqdm import tqdm

# Parameters
PATCH_SIZE = 250
RGB_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/Subotica_Deo.png"
DEM_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/DEM.png"
DSM_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/DSM.png"
MASK_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/mask.png"
RGB_OUTPUT_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/RGB images"
DEM_OUTPUT_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/DEM images"
DSM_OUTPUT_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/DSM images"
MASK_OUTPUT_DIR = "/home/igor/Projects/Geoinformatika/Data/Dataset/Masks"

# Function to create patches
def create_patches(image_path, output_dir, patch_size):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Calculate grid size
    grid_rows = height // patch_size
    grid_cols = width // patch_size
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through rows and columns to extract patches
    for i in tqdm(range(grid_rows), desc=f"Processing {os.path.basename(image_path)}"):
        for j in range(grid_cols):
            # Calculate patch coordinates
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size

            # Extract patch
            patch = image[y_start:y_end, x_start:x_end]

            # Save patch
            patch_filename = f"patch_{i}_{j}.png"
            patch_path = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_path, patch)

# Create dataset
create_patches(RGB_IMAGE_PATH, RGB_OUTPUT_DIR, PATCH_SIZE)
create_patches(DEM_IMAGE_PATH, DEM_OUTPUT_DIR, PATCH_SIZE)
create_patches(DSM_IMAGE_PATH, DSM_OUTPUT_DIR, PATCH_SIZE)
create_patches(MASK_IMAGE_PATH, MASK_OUTPUT_DIR, PATCH_SIZE)
