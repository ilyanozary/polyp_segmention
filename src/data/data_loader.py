import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_dir(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_and_resize_images(image_paths, size=(224, 224), is_mask=False):
    """
    Load and resize images or masks
    
    Args:
        image_paths (list): List of image paths
        size (tuple): Target size for resizing (height, width)
        is_mask (bool): Whether the images are masks
    
    Returns:
        numpy.ndarray: Array of processed images/masks
    """
    images = []
    for path in tqdm(image_paths, desc="Resizing images" if not is_mask else "Resizing masks"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        img = cv2.resize(img, size)
        if not is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.expand_dims(img, axis=-1)
        images.append(img)
    return np.array(images, dtype='float32')

def load_data(path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load dataset and split into train, validation, and test sets
    
    Args:
        path (str): Path to the dataset directory
        test_size (float): Proportion of dataset to include in test split
        val_size (float): Proportion of dataset to include in validation split
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_data, val_data, test_data) where each is a tuple of (X, y)
    """
    # Load image and mask paths
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.jpg")))

    # Split into train and temp
    train_x, temp_x, train_y, temp_y = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    # Split temp into validation and test
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=random_state
    )

    # Resize images and masks
    train_x = load_and_resize_images(train_x)
    val_x = load_and_resize_images(val_x)
    test_x = load_and_resize_images(test_x)

    train_y = load_and_resize_images(train_y, is_mask=True)
    val_y = load_and_resize_images(val_y, is_mask=True)
    test_y = load_and_resize_images(test_y, is_mask=True)

    # Normalize to [0, 1]
    train_x = train_x / 255.0
    val_x = val_x / 255.0
    test_x = test_x / 255.0

    train_y = train_y / 255.0
    val_y = val_y / 255.0
    test_y = test_y / 255.0

    return (train_x, train_y), (val_x, val_y), (test_x, test_y) 