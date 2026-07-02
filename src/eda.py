import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from preprocessing import CLASS_NAMES

def main():
    print("[INFO] Starting Exploratory Data Analysis (EDA)...")
    
    raw_dir = Path("data/raw")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Class Distribution Analysis ──────────────────────────────────
    print("[INFO] Analyzing class distribution...")
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_path = raw_dir / class_name
        if class_path.exists():
            files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
            class_counts[class_name] = len(files)
        else:
            class_counts[class_name] = 0
            
    df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    
    # Plot Class Distribution
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Class", y="Count", data=df_counts, palette="viridis")
    plt.title("Class Distribution in Sentinel-2 & Desert Dataset", fontsize=16, fontweight="bold")
    plt.xlabel("Land Type Class", fontsize=12)
    plt.ylabel("Number of Image Patches", fontsize=12)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight="bold")
                    
    plt.tight_layout()
    plt.savefig(outputs_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f" - Saved class distribution plot to: {outputs_dir / 'class_distribution.png'}")
    
    # ── 2. Pixel Value Distribution Histograms ──────────────────────────
    print("[INFO] Plotting pixel value distributions per class (this might take a few seconds)...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    num_samples = 100  # Number of random samples to compute histogram per class
    np.random.seed(42)
    
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = raw_dir / class_name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        
        # Subsample for efficiency
        if len(images) > num_samples:
            sampled_images = np.random.choice(images, num_samples, replace=False)
        else:
            sampled_images = images
            
        all_pixels = []
        for img_path in sampled_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_pixels.append(img.reshape(-1, 3))
                
        if len(all_pixels) > 0:
            pixels = np.concatenate(all_pixels, axis=0)
            
            # Plot histograms for Red, Green, Blue channels
            colors = ['red', 'green', 'blue']
            for c, color in enumerate(colors):
                axes[idx].hist(pixels[:, c], bins=50, color=color, 
                              alpha=0.4, density=True, label=color.upper())
                              
            axes[idx].set_title(f"{class_name} (Sample size: {len(sampled_images)})", fontsize=14, fontweight="bold")
            axes[idx].set_xlabel("Pixel Intensity Value", fontsize=11)
            axes[idx].set_ylabel("Density", fontsize=11)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        else:
            axes[idx].text(0.5, 0.5, "No Images Found", ha='center', va='center')
            axes[idx].set_title(class_name)
            
    plt.suptitle("RGB Pixel Value Distributions by Land Type Class", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outputs_dir / "pixel_distributions.png", dpi=150)
    plt.close()
    print(f" - Saved pixel distribution plot to: {outputs_dir / 'pixel_distributions.png'}")
    
    # ── 3. Color Channel Correlations ─────────────────────────────────
    print("[INFO] Computing color channel correlation matrix...")
    all_sampled_pixels = []
    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        if len(images) > 20:
            sampled = np.random.choice(images, 20, replace=False)
            for img_path in sampled:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    all_sampled_pixels.append(img.reshape(-1, 3))
                    
    if len(all_sampled_pixels) > 0:
        pixels_stack = np.concatenate(all_sampled_pixels, axis=0)
        df_pixels = pd.DataFrame(pixels_stack, columns=["Red", "Green", "Blue"])
        corr_matrix = df_pixels.corr()
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".4f", square=True)
        plt.title("RGB Channel Correlation Matrix (Across All Classes)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(outputs_dir / "channel_correlations.png", dpi=150)
        plt.close()
        print(f" - Saved channel correlation matrix plot to: {outputs_dir / 'channel_correlations.png'}")
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.to_string())
    else:
        print("[WARNING] Could not compute correlation matrix (no pixels).")
        
    print("\n[INFO] EDA completed successfully!")

if __name__ == "__main__":
    main()
