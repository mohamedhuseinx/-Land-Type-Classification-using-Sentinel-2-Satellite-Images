import os
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from data_loading import get_dataloaders
from preprocessing import CLASS_NAMES

def main():
    print("[INFO] Starting verification of Milestone 1...")
    
    # 1. Verify file structure
    raw_dir = Path("data/raw")
    splits_dir = Path("data/splits")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Verifying directory structure:")
    print(f" - Raw directory exists: {raw_dir.exists()}")
    print(f" - Splits directory exists: {splits_dir.exists()}")
    
    # 2. Verify class balance in raw directory
    print("\n2. Checking image counts per class in data/raw:")
    total_raw = 0
    for class_name in CLASS_NAMES:
        class_path = raw_dir / class_name
        if class_path.exists():
            count = len(list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")))
            print(f"   - {class_name}: {count} images")
            total_raw += count
        else:
            print(f"   - [ERROR] Directory for class {class_name} is missing!")
    print(f" Total raw images found: {total_raw}")
    
    # 3. Verify splits CSV files
    print("\n3. Checking splits files:")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    
    for name, path in [("Train", train_csv), ("Validation", val_csv), ("Test", test_csv)]:
        if path.exists():
            df = pd.read_csv(path)
            print(f"   - {name} split ({path.name}): {len(df)} samples")
            # Verify file paths exist
            missing_files = 0
            for idx, row in df.head(10).iterrows():
                if not Path(row["filepath"]).exists():
                    missing_files += 1
            if missing_files > 0:
                print(f"     [WARNING] Checked first 10 paths and {missing_files} were missing!")
            else:
                print(f"     First 10 sample paths verified successfully.")
        else:
            print(f"   - [ERROR] {name} CSV is missing!")

    # 4. Verify PyTorch DataLoader loading
    print("\n4. Verifying PyTorch DataLoaders:")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            str(train_csv),
            str(val_csv),
            str(test_csv),
            batch_size=8
        )
        
        # Get one batch
        images, labels = next(iter(train_loader))
        
        print(f"   - Successful batch extraction!")
        print(f"   - Images shape: {images.shape} (Expected: torch.Size([8, 3, 64, 64]))")
        print(f"   - Labels shape: {labels.shape} (Expected: torch.Size([8]))")
        print(f"   - Labels in batch: {labels.tolist()}")
        print(f"   - Image min/max/mean: {images.min().item():.4f} / {images.max().item():.4f} / {images.mean().item():.4f}")
        
        # 5. Visualize and save a sample plot
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        # Helper to unnormalize image for plotting
        # Normalized = (raw - mean) / std => raw = Normalized * std + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for i in range(8):
            img_tensor = images[i].permute(1, 2, 0).numpy() # Shape: (64, 64, 3)
            # Unnormalize
            img_unnorm = img_tensor * std + mean
            img_unnorm = np.clip(img_unnorm, 0, 1)
            
            axes[i].imshow(img_unnorm)
            label_idx = labels[i].item()
            axes[i].set_title(CLASS_NAMES[label_idx], fontsize=12, fontweight="bold")
            axes[i].axis("off")
            
        plt.suptitle("Milestone 1 Verification Batch (Augmented & Normalised)", fontsize=16)
        plt.tight_layout()
        plot_path = outputs_dir / "verification_batch.png"
        plt.savefig(plot_path, dpi=150)
        print(f"   - Saved verification plot to: {plot_path.absolute()}")
        
    except Exception as e:
        print(f"   - [ERROR] DataLoader verification failed: {e}")

if __name__ == "__main__":
    main()
