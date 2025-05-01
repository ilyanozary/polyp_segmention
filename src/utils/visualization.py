import matplotlib.pyplot as plt
import numpy as np

def display_image_mask_pairs(images, masks, title, num_images=10):
    """
    Display a set of images with their corresponding masks
    
    Args:
        images (numpy.ndarray): Array of images
        masks (numpy.ndarray): Array of masks
        title (str): Title for the plot
        num_images (int): Number of image-mask pairs to display
    """
    plt.figure(figsize=(5, 5))
    
    for i in range(min(num_images, len(images))):
        img = images[i]
        mask = masks[i]
        
        # Plot original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{title} - Image {i+1}")
        
        # Plot mask
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(mask.squeeze(), cmap="gray")
        plt.axis('off')
        plt.title(f"{title} - Mask {i+1}")
    
    plt.tight_layout()
    plt.show()

def display_predictions(images, true_masks, predicted_masks, num_images=5):
    """
    Display original images, true masks, and predicted masks
    
    Args:
        images (numpy.ndarray): Array of original images
        true_masks (numpy.ndarray): Array of true masks
        predicted_masks (numpy.ndarray): Array of predicted masks
        num_images (int): Number of samples to display
    """
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis('off')
        
        plt.subplot(3, num_images, i + num_images + 1)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')
        
        plt.subplot(3, num_images, i + 2 * num_images + 1)
        plt.imshow(predicted_masks[i].squeeze() > 0.5, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show() 