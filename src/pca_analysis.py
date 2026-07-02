import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CLASS_NAMES = ["Agriculture", "Water", "Urban", "Desert", "Roads", "Trees"]

def load_sample_images(data_dir, class_names, samples_per_class=200, image_size=64):
    images = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = Path(data_dir) / class_name
        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        np.random.seed(42)
        selected = np.random.choice(files, min(samples_per_class, len(files)), replace=False)
        for f in tqdm(selected, desc=f"Loading {class_name}"):
            img = cv2.imread(str(f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def apply_pca(images, n_components=50):
    N = images.shape[0]
    flat = images.reshape(N, -1).astype(np.float32)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    pca = PCA(n_components=n_components, random_state=42)
    features = pca.fit_transform(flat_scaled)
    return features, pca, scaler

def plot_explained_variance(pca, save_path="outputs/pca_explained_variance.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_*100, color='steelblue', alpha=0.7, label='Individual')
    axes[0].plot(range(1, len(cumvar)+1), cumvar, 'ro-', label='Cumulative')
    axes[0].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Explained Variance Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    dims_90 = np.searchsorted(cumvar, 90) + 1
    axes[1].plot(range(1, min(11, len(cumvar)+1)), cumvar[:min(10, len(cumvar))], 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].set_title(f'First 10 Components (90% at {dims_90} components)')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('PCA Analysis of Satellite Image Patches', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved explained variance plot to {save_path}")
    print(f"Components needed for 90% variance: {dims_90}")
    return dims_90

def plot_pca_scatter(features, labels, save_path="outputs/pca_scatter.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(CLASS_NAMES)))
    for label in range(len(CLASS_NAMES)):
        mask = labels == label
        axes[0].scatter(features[mask, 0], features[mask, 1], c=[colors[label]], label=CLASS_NAMES[label], alpha=0.5, s=5)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA: First 2 Components', fontsize=14, fontweight='bold')
    axes[0].legend(markerscale=5)
    axes[0].grid(True, alpha=0.3)
    for label in range(len(CLASS_NAMES)):
        mask = labels == label
        axes[1].scatter(features[mask, 2], features[mask, 3], c=[colors[label]], label=CLASS_NAMES[label], alpha=0.5, s=5)
    axes[1].set_xlabel('PC3')
    axes[1].set_ylabel('PC4')
    axes[1].set_title('PCA: Components 3 & 4', fontsize=14, fontweight='bold')
    axes[1].legend(markerscale=5)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('PCA Class Separability Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PCA scatter plot to {save_path}")

def main():
    print("=" * 60)
    print("  PCA ANALYSIS — Sentinel-2 Land Type Classification")
    print("=" * 60)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    images, labels = load_sample_images("data/raw", CLASS_NAMES, samples_per_class=200)
    print(f"\nLoaded {len(images)} images with shape {images.shape}")
    features, pca, scaler = apply_pca(images, n_components=50)
    dims_90 = plot_explained_variance(pca)
    plot_pca_scatter(features, labels)
    print("\nPCA Analysis Complete!")

if __name__ == "__main__":
    main()
