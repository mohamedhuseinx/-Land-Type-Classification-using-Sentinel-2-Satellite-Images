import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Target image size for training
IMAGE_SIZE = 64

class LandTypeDataset(Dataset):
    """Custom PyTorch Dataset for Land Type Classification."""
    
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str or Path): Path to the split CSV file (train.csv, val.csv, test.csv).
            transform (albumentations.Compose, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])
        
        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Albumentations transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, label

# ── Data Augmentation Pipelines ────────────────────────────────────

def get_train_transforms(image_size=IMAGE_SIZE):
    """
    Get training augmentation pipeline.
    Satellite images are rotation-invariant, so aggressive geometric
    transforms (rotations, flips) are appropriate.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
            std=[0.229, 0.224, 0.225]    # Standard ImageNet stds
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        ToTensorV2()                     # Convert to torch tensor (C, H, W)
    ])

def get_val_transforms(image_size=IMAGE_SIZE):
    """Get validation/test pipeline: only normalization and resizing, no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# ── DataLoader Builder ──────────────────────────────────────────────

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=64, num_workers=4, pin_memory=True):
    """
    Build and return PyTorch DataLoaders for train, validation, and test splits.
    """
    train_dataset = LandTypeDataset(train_csv, transform=get_train_transforms())
    val_dataset = LandTypeDataset(val_csv, transform=get_val_transforms())
    test_dataset = LandTypeDataset(test_csv, transform=get_val_transforms())
    
    # On Windows, num_workers > 0 can sometimes cause issues in multiprocessing,
    # so we default to 0 if we encounter issues, but num_workers=0 is safe.
    # We will use num_workers=0 to ensure cross-platform safety by default on Windows.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test loading
    train_loader, val_loader, test_loader = get_dataloaders(
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        batch_size=8
    )
    
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expected: [8, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")  # Expected: [8]
    print(f"Labels: {labels}")
