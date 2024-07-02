# Importing libraries
import os
import cv2
import numpy as np

# Parameters
MASKS_DIR_PATH = "/home/igor/Projects/Geoinformatika/Data/Mask data/Mask patches"
OUTPUT_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/mask.png"
WIDTH = 6750
HEIGHT = 6000
INPUT_GRID = (4, 4)


# Calculate the size of the recomposed image
total_height = HEIGHT * INPUT_GRID[0]
total_width = WIDTH * INPUT_GRID[1]

# Create an empty canvas for the output image
output_image = np.zeros((total_height, total_width), dtype=np.uint8)

# Iterate through rows and columns to place patches
for i in range(INPUT_GRID[0]):
    for j in range(INPUT_GRID[1]):
        # Calculate patch coordinates
        y_start = i * HEIGHT
        y_end = (i + 1) * HEIGHT
        x_start = j * WIDTH
        x_end = (j + 1) * WIDTH

        # Patch filename
        patch_filename = f"mask_patch_{i}_{j}.png"
        patch_path = os.path.join(MASKS_DIR_PATH, patch_filename)

        # Check if patch exists and read it
        if os.path.exists(patch_path):
            patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create a black patch if it does not exist
            patch = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            print(f"Patch {patch_filename} does not exist. Filling with black.")

        # Place the patch in the output image
        output_image[y_start:y_end, x_start:x_end] = patch

# Save the recomposed image
cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)

print("Image recomposed from patches successfully.")
