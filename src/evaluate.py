import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loading import get_dataloaders, get_val_transforms, LandTypeDataset
from model import CustomCNN, build_resnet50, CLASS_NAMES, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


def load_model(arch="resnet50", checkpoint_path=None):
    if arch == "custom_cnn":
        model = CustomCNN()
    elif arch == "resnet50":
        model = build_resnet50(freeze_base=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path="outputs/evaluation/confusion_matrix.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curves(y_true, y_probs, save_path="outputs/evaluation/roc_curves.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, 'k--', linewidth=2.5, label=f'Macro Average (AUC = {macro_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('Per-Class ROC Curves', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")
    return macro_auc


def plot_training_history(history, save_path="outputs/evaluation/training_history.png"):
    if history is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {save_path}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_target_layer(model):
    if hasattr(model, 'layer4'):
        return model.layer4[2].conv3
    else:
        return model.features[-3]

def plot_gradcam_samples(model, loader, save_path="outputs/evaluation/gradcam.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    images, labels = next(iter(loader))
    n_samples = min(4, len(images))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    for i in range(n_samples):
        img = images[i].cpu()
        label = labels[i].item()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img * std + mean
        img_display = torch.clamp(img_display, 0, 1).permute(1, 2, 0).numpy()
        cam, pred_class = gradcam.generate(images[i:i+1].to(device))
        import cv2
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[2]))
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"True: {CLASS_NAMES[label]}", fontsize=10)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(cam_resized, cmap='jet')
        axes[i, 1].set_title(f"Grad-CAM", fontsize=10)
        axes[i, 1].axis('off')
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * img_display + 0.4 * heatmap / 255.0)
        overlay = np.clip(overlay, 0, 1)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Pred: {CLASS_NAMES[pred_class]}", fontsize=10)
        axes[i, 2].axis('off')
        axes[i, 3].axis('off')
    plt.suptitle("Grad-CAM: Model Attention Maps", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM to {save_path}")


def main():
    print("="*60)
    print("  MODEL EVALUATION — Sentinel-2 Land Type Classification")
    print("="*60)
    Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet50", choices=["custom_cnn", "resnet50"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--history", type=str, default=None, help="Path to history npy file")
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders(
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        batch_size=64, num_workers=0,
    )
    model = load_model(args.arch, args.checkpoint)
    y_true, y_pred, y_probs = get_predictions(model, test_loader)
    accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")

    plot_confusion_matrix(y_true, y_pred)
    macro_auc = plot_roc_curves(y_true, y_probs)

    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    if args.history:
        history = np.load(args.history, allow_pickle=True).item()
        plot_training_history(history)

    try:
        train_loader, _, _ = get_dataloaders(
            "data/splits/train.csv",
            "data/splits/val.csv",
            "data/splits/test.csv",
            batch_size=8, num_workers=0,
        )
        plot_gradcam_samples(model, train_loader)
    except Exception as e:
        print(f"[WARNING] Grad-CAM failed: {e}")

    results = {
        "architecture": args.arch,
        "test_accuracy": float(accuracy),
        "macro_auc": float(macro_auc),
    }
    import json
    with open("outputs/evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/evaluation/results.json")
    print(f"[RESULT] Test Accuracy: {accuracy*100:.2f}%")
    print(f"[RESULT] Macro AUC: {macro_auc:.4f}")

if __name__ == "__main__":
    main()
