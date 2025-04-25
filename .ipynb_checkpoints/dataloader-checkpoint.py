import os
import numpy as np
import tensorflow as tf
import albumentations as A
from albumentations.core.composition import OneOf
from glob import glob
import cv2

class PolypDatasetTF(tf.keras.utils.Sequence):
    def __init__(self, images_dir, masks_dir, batch_size=8, image_size=(384, 384), augment=False):
        self.image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, "*_mask.png")))
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment

        if augment:
            self.transform = self.get_augmentations()
            self.aug_repeat = 3
            print(f"[!] Augmentation is enabled. Every image is augmented ×{self.aug_repeat}.")
        else:
            self.transform = None
            self.aug_repeat = 1

        self.dataset_length = len(self.image_paths) * self.aug_repeat
        print(f"✅ {len(self.image_paths)}Initial image → {self.dataset_length} Image after data augmentation")

    def __len__(self):
        return self.dataset_length // self.batch_size

    def __getitem__(self, idx):
        images = []
        masks = []
    
        for i in range(self.batch_size):
            base_idx = (idx * self.batch_size + i) % len(self.image_paths)
            img_path = self.image_paths[base_idx]
            mask_path = self.mask_paths[base_idx]
    
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
            # ✅ Resize both before augmentation to avoid size mismatch
            img = cv2.resize(img, self.image_size)
            mask = cv2.resize(mask, self.image_size)
    
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
    
            img = img.astype(np.float32) / 255.0
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
    
            images.append(img)
            masks.append(mask)
    
        return np.array(images), np.array(masks)

    def get_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        ])

if __name__ == "__main__":
    train_dataset = PolypDatasetTF(
        images_dir="Dataset700_label/images",
        masks_dir="Dataset700_label/predicted_masks",
        batch_size=8,
        augment=True
    )

    for images, masks in train_dataset:
        print("🔹 Batch images:", images.shape)
        print("🔹 Batch masks :", masks.shape)
        print("🔹 Images range:", images.min(), "-", images.max())
        print("🔹 Masks unique :", np.unique(masks))
        break
