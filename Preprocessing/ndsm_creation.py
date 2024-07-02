# Importing libraries
import numpy as np
from PIL import Image

# Parameters
DEM_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/DEM.png"
DSM_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/DSM.png"
NDSM_IMAGE_PATH = "/home/igor/Projects/Geoinformatika/Data/Raw data/nDSM_binarized.png"
THRESHOLD = 3

# Increase the decompression bomb limit
Image.MAX_IMAGE_PIXELS = None

# Open DEM, DSM images
dem_image = Image.open(DEM_IMAGE_PATH)
dsm_image = Image.open(DSM_IMAGE_PATH)

# Convert images to grayscale
dem_array = np.array(dem_image.convert('L'))
dsm_array = np.array(dsm_image.convert('L'))

# Calculate NDSM
ndsm_array = dsm_array - dem_array

# Thresholding
ndsm_array = np.where(ndsm_array > THRESHOLD, 255, 0).astype(np.uint8)

# Convert array back to image
ndsm_image = Image.fromarray(ndsm_array)

# Save NDSM image
ndsm_image.save(NDSM_IMAGE_PATH)
