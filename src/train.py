import os
import numpy as np
import tensorflow as tf
from data.data_loader import load_data
from models.deeplabv3_plus import deeplabv3_plus
from metrics.metrics import (
    accuracy, precision, recall, dice_coefficient, iou, mixed_loss
)
from utils.visualization import display_image_mask_pairs, display_predictions

def train_model(data_path, model_save_path, batch_size=4, epochs=10):
    """
    Train the DeepLabV3+ model
    
    Args:
        data_path (str): Path to the dataset
        model_save_path (str): Path to save the trained model
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load dataset
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(data_path)

    # Display sample images
    display_image_mask_pairs(train_x, train_y, "Training Set", 2)
    display_image_mask_pairs(val_x, val_y, "Validation Set", 2)
    display_image_mask_pairs(test_x, test_y, "Testing Set", 2)

    # Create model
    model = deeplabv3_plus((224, 224, 3))

    # Learning rate schedule
    initial_learning_rate = 0.0005
    decay_steps = 1000
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
    )

    # Compile model
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=mixed_loss,
        metrics=[accuracy, precision, recall, dice_coefficient, iou]
    )

    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_x, val_y),
        verbose=1
    )

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    # Display predictions on test set
    predictions = model.predict(test_x[:5])
    display_predictions(test_x[:5], test_y[:5], predictions)

if __name__ == "__main__":
    data_path = "Kvasir-SEG/Kvasir-SEG"
    model_save_path = "models/deeplabv3_plus_mobilenet_kvasir.h5"
    train_model(data_path, model_save_path) 