import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    """Intersection over Union (IoU) metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + K.epsilon())

def dice_coefficient(y_true, y_pred):
    """Dice Coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric"""
    true_positive = K.sum(K.round(y_true * y_pred))
    predicted_positive = K.sum(K.round(y_pred))
    return true_positive / (predicted_positive + K.epsilon())

def recall(y_true, y_pred):
    """Recall metric"""
    true_positive = K.sum(K.round(y_true * y_pred))
    possible_positive = K.sum(K.round(y_true))
    return true_positive / (possible_positive + K.epsilon())

def accuracy(y_true, y_pred):
    """Binary Accuracy metric"""
    correct_predictions = K.sum(K.cast(K.equal(K.round(y_true), K.round(y_pred)), dtype="float32"))
    total_predictions = K.cast(K.prod(K.shape(y_true)), dtype="float32")
    return correct_predictions / (total_predictions + K.epsilon())

def focal_loss(alpha=0.8, gamma=2.0):
    """Focal Loss for handling class imbalance"""
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
        focal_weight = alpha * K.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        return K.mean(focal_weight * K.binary_crossentropy(y_true, y_pred))
    return loss

def dice_loss(y_true, y_pred):
    """Dice Loss for segmentation tasks"""
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mixed_loss(y_true, y_pred):
    """Combined Focal Loss and Dice Loss"""
    return focal_loss()(y_true, y_pred) + dice_loss(y_true, y_pred) 