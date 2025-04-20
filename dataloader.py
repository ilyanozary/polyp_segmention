import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, 
    Resize, 
    HorizontalFlip, 
    VerticalFlip, 
    RandomRotate90,
    RandomBrightnessContrast,
    GaussNoise,
    ColorJitter,
    Normalize,
    RandomCrop
)
from albumentations.pytorch import ToTensorV2

class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mode='train'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mode = mode

        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('_mask.png')])

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must match")

        if transform is None:
            if mode == 'train':
                self.transform = self.get_train_transform()
            else:
                self.transform = self.get_val_transform()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '_mask.png'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # اگر اندازه‌ها برابر نباشن، ماسک رو به اندازه تصویر تغییر بده
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # ماسک باینری
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
        else:
            mask = (mask > 0).astype(np.float32)

        return image, mask

    @staticmethod
    def get_train_transform():
        return Compose([
            RandomCrop(height=384, width=384),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.2),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_val_transform():
        return Compose([
            Resize(384, 384),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_dataloader(images_dir, masks_dir, batch_size=8, mode='train', num_workers=4):
    """
    Create DataLoader for polyp dataset
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to masks folder
        batch_size (int): batch size
        mode (str): 'train' or 'val' or 'test'
        num_workers (int): number of workers for data loading
        
    Returns:
        DataLoader: ready to use dataloader
    """
    dataset = PolypDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode=mode
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

# Usage example
if __name__ == '__main__':
    # Change paths according to your project structure
    images_dir = "Dataset700_label/images"
    masks_dir = "Dataset700_label/predicted_masks"
    
    # Create dataloader
    train_loader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=8,
        mode='train'
    )
    
    # Test dataloader
    for images, masks in train_loader:
        print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
        print(f"Value ranges - Images: [{images.min():.3f}, {images.max():.3f}], Masks: [{masks.min():.3f}, {masks.max():.3f}]")
        break