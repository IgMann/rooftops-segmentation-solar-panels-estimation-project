# Importing libraries
import os
import cv2

# Parameters
IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/nDSM_binarized.png"
OUTPUT_PATH = "/home/igor/Projects/Geoinformatika/Data/Mask data/nDSM patches"
OUTPUT_GRID = (4, 4)

# Load the image
image = cv2.imread(IMAGE_PATH)

# Get image dimensions
height, width, _ = image.shape

# Calculate patch size
patch_height = height // OUTPUT_GRID[0]
patch_width = width // OUTPUT_GRID[1]

# Check if patch size is round number
if patch_height * OUTPUT_GRID[0] != height or patch_width * OUTPUT_GRID[1] != width:
    print("Error: Patch size is not a round number. Aborting execution.")
    exit()

# Ensure the patch size is an integer
patch_height = int(patch_height)
patch_width = int(patch_width)

# Iterate through rows and columns to extract patches
for i in range(OUTPUT_GRID[0]):
    for j in range(OUTPUT_GRID[1]):
        # Calculate patch coordinates
        y_start = i * patch_height
        y_end = (i + 1) * patch_height
        x_start = j * patch_width
        x_end = (j + 1) * patch_width

        # Extract patch
        patch = image[y_start:y_end, x_start:x_end]

        # Save patch
        patch_filename = f"patch_{i}_{j}.png"
        patch_path = os.path.join(OUTPUT_PATH, patch_filename)
        cv2.imwrite(patch_path, patch)

print("Image decomposed into patches successfully.")
