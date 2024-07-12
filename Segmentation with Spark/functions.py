# Importing Libraries
import os
import io
import shutil
import math
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import cv2
from PIL import Image

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, concat, lit, array
from pyspark.sql.types import StringType, StructType, StructField, DoubleType, ArrayType, BinaryType

from pyspark.ml.evaluation import BinaryClassificationEvaluator

def read_image_file(path):
    """
    Reads an image file and returns its binary content.

    Args:
        path (str): The file path of the image to read.

    Returns:
        bytes: The binary content of the image file if it exists.
        None: If the file does not exist, returns None.
    """
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    criterion : torch.nn.modules.loss._Loss
        The loss function to be used.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for updating model parameters.
    device : torch.device
        The device on which to perform training (e.g., 'cpu' or 'cuda').

    Returns
    -------
    float
        The average loss over the training dataset.
    """
    model.train()
    running_loss = 0.0
    for inputs, masks, _ in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)

def evaluate(model, val_loader, device, spark):
    """
    Evaluate the model on the validation dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    device : torch.device
        The device on which to perform evaluation (e.g., 'cpu' or 'cuda').
    spark : pyspark.sql.SparkSession
        The Spark session used for processing.

    Returns
    -------
    float
        The area under the ROC curve (AUC-ROC) for the model predictions.
    """
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for inputs, masks, _ in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())
    
    # Convert predictions and masks to Spark DataFrame
    schema = StructType([
        StructField("prediction", DoubleType(), False),
        StructField("mask", DoubleType(), False)
    ])
    data = [(float(pred), float(mask)) for pred, mask in zip(all_preds, all_masks)]
    spark_df = spark.createDataFrame(data, schema)

    # Calculate Jaccard score using Spark
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="prediction", 
        labelCol="mask", 
        metricName="areaUnderROC"
    )
    auc_roc = evaluator.evaluate(spark_df)

    return auc_roc

def save_predictions(model, data_loader, device, result_dir):
    """
    Save model predictions as images.

    Parameters
    ----------
    model : torch.nn.Module
        The model used for making predictions.
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the dataset for which predictions are to be made.
    device : torch.device
        The device on which to perform inference (e.g., 'cpu' or 'cuda').
    result_dir : str
        Directory where the prediction images will be saved.

    Returns
    -------
    None
    """
    model.eval()
    total_images = sum(len(batch[2]) for batch in data_loader)
    
    with torch.no_grad():
        with tqdm(total=total_images, desc="Saving predictions") as pbar:
            for inputs, _, image_names in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                for pred, image_name in zip(preds, image_names):
                    pred_image = pred.squeeze().cpu().numpy() * 255
                    pred_image = Image.fromarray(pred_image.astype(np.uint8))
                    pred_image.save(os.path.join(result_dir, image_name))
                    pbar.update(1)

def conditional_median_filter(image):
    """
    Apply a conditional median filter to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to be filtered.

    Returns
    -------
    numpy.ndarray
        The filtered image.
    """
    padded_image = np.pad(image, pad_width=1, mode="constant", constant_values=0)
    result = np.copy(image)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            window = padded_image[i-1:i+2, j-1:j+2]
            center_pixel = window[1, 1]
            if center_pixel == 0 and np.all(window[window != center_pixel] == 255):
                result[i-1, j-1] = 255
            elif center_pixel == 255 and np.all(window[window != center_pixel] == 0):
                result[i-1, j-1] = 0

    return result

def count_large_objects(image_path, percentage_threshold=0):
    """
    Count the number of large objects in a binary image based on a given percentage threshold.

    Parameters
    ----------
    image_path : str
        Path to the binary image file.
    percentage_threshold : float, optional
        Minimum percentage area of the image that an object must occupy to be considered large (default is 0).

    Returns
    -------
    int
        The count of large objects in the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_pixels = image.size
    threshold_pixels = (percentage_threshold / 100) * total_pixels

    large_objects_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold_pixels:
            large_objects_count += 1
    
    return large_objects_count

def find_largest_objects_areas(image_path, N, square_pixel_area):
    """
    Find the areas of the N largest objects in a binary image and convert them to square meters.

    Parameters
    ----------
    image_path : str
        Path to the binary image file.
    N : int
        Number of largest objects to find.
    square_pixel_area : float
        Area of a single pixel in square meters.

    Returns
    -------
    list of float
        A list of areas of the N largest objects in square meters, rounded to two decimal places.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    largest_areas = sorted(areas, reverse=True)[:N]

    largest_areas_in_square_meters = [round(area * square_pixel_area, 2) for area in largest_areas]

    return largest_areas_in_square_meters

def roof_area_calculation(object_area, angle_degrees=30):
    """
    Calculate the roof area based on the projected area and the roof angle.

    Parameters
    ----------
    object_area : float
        The projected area of the object (e.g., roof) in square meters.
    angle_degrees : float, optional
        The angle of the roof in degrees (default is 30).

    Returns
    -------
    float
        The actual roof area in square meters, rounded to two decimal places.
    """
    angle_radians = math.radians(angle_degrees)
    projection_factor = 1 / math.cos(angle_radians)
    roof_area = round(object_area * projection_factor, 2)
    
    return roof_area

def solar_panels_estimation(roof_area, solar_panel_area, efficiency_factor=0.90):
    """
    Estimate the number of solar panels that can fit on a roof area.

    Parameters
    ----------
    roof_area : float
        The total roof area in square meters.
    solar_panel_area : float
        The area of a single solar panel in square meters.
    efficiency_factor : float, optional
        Efficiency factor to account for non-usable space (default is 0.90).

    Returns
    -------
    int
        The estimated number of solar panels that can fit on the roof.
    """
    usable_area = roof_area * efficiency_factor
    number_of_panels = round(usable_area / solar_panel_area)
    
    return number_of_panels

def calculate_iou(mask_path, result_path):
    """
    Calculate the Intersection over Union (IoU) between two binary images.

    Parameters
    ----------
    mask_path : str
        Path to the ground truth binary mask image file.
    result_path : str
        Path to the predicted binary mask image file.

    Returns
    -------
    float
        The IoU score as a percentage, rounded to two decimal places.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None or result is None:
        return 0.0
    
    mask = mask / 255
    result = result / 255
    intersection = np.logical_and(mask, result).sum()
    union = np.logical_or(mask, result).sum()
    iou = intersection / union if union != 0 else 0
    
    return round(iou * 100, 2)

