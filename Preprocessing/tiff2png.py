# Importing libraries
from PIL import Image

# Parameters
TIFF_FILE = "/home/igor/Projects/Geoinformatika/Data/Raw data/Subotica_Deo.tif"
PNG_FILE = "/home/igor/Projects/Geoinformatika/Data/Raw data/Subotica_Deo.png"

# Increase the decompression bomb limit
Image.MAX_IMAGE_PIXELS = None

# Open the TIFF image
tiff_image = Image.open(TIFF_FILE)

# Convert to RGB mode
rgb_image = tiff_image.convert("RGB")

# Save the image as PNG
rgb_image.save(PNG_FILE, format="PNG")

print(f"Image converted and saved to {PNG_FILE}")
