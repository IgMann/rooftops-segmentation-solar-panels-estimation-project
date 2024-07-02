# Importing libraries
import json
import os
from tqdm import tqdm
import numpy as np
import cv2

# Parameters
COCO_JSON_PATH = "/home/igor/MEGAsync/Faks/Geoprostorni sistemi/Projekat/Preprocessing/result.json"
MASKS_DIR_PATH = "/home/igor/Projects/Geoinformatika/Data/Mask data/Mask patches"
WIDTH = 6750
HEIGHT = 6000

# Load COCO JSON file
with open(COCO_JSON_PATH, "r") as file:
    coco_data = json.load(file)

# Create masks directory if it doesn't exist
if not os.path.exists(MASKS_DIR_PATH):
    os.makedirs(MASKS_DIR_PATH)

# Function to create a mask from segmentation data
def create_mask(image_info, annotations):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            segmentation = ann['segmentation']
            for seg in segmentation:
                pts = np.array(seg, np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], color=1)
    return mask

# Iterate through images and save masks
for image in tqdm(coco_data["images"], desc="Processing images"):
    image_file_name = os.path.basename(image["file_name"])
    mask_file_name = f"mask_{image_file_name.split('-')[-1]}"
    
    # Get annotations for the current image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
    
    # Create the mask
    mask = create_mask(image, annotations)
    
    # Save the mask
    mask_path = os.path.join(MASKS_DIR_PATH, mask_file_name)
    cv2.imwrite(mask_path, mask * 255)  # Save mask as an image
