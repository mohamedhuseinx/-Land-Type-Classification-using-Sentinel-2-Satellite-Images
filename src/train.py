import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loading import get_dataloaders
from model import CustomCNN, build_resnet50, CLASS_NAMES, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=NUM_CLASSES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[Val]", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, weight_decay=1e-4,
                label_smoothing=0.1, model_name="model", checkpoint_dir="models/checkpoints"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    criterion = LabelSmoothingLoss(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_dir}/{model_name}_best.pth")
            print(f"  -> New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"[Early Stopping] No improvement for 10 epochs. Best Val Acc: {best_val_acc:.2f}%")
                break

    writer.close()
    print(f"\n[INFO] Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return history, best_val_acc


def train_two_phase(train_loader, val_loader, architecture="resnet50",
                    phase1_epochs=15, phase2_epochs=50):
    if architecture == "custom_cnn":
        print("\n" + "="*60)
        print("  BUILDING CUSTOM CNN BASELINE")
        print("="*60)
        model = CustomCNN().to(device)
        print(model)
        history, best_acc = train_model(
            model, train_loader, val_loader,
            epochs=phase1_epochs + phase2_epochs,
            lr=1e-3, model_name="CustomCNN",
        )
        return model, history, best_acc

    elif architecture == "resnet50":
        print("\n" + "="*60)
        print("  PHASE 1: RESNET50 HEAD TRAINING (BASE FROZEN)")
        print("="*60)
        model = build_resnet50(freeze_base=True).to(device)
        history1, _ = train_model(
            model, train_loader, val_loader,
            epochs=phase1_epochs, lr=1e-3,
            model_name="ResNet50_phase1",
        )

        print("\n" + "="*60)
        print("  PHASE 2: RESNET50 FINE-TUNING (LAYERS 5+ UNFROZEN)")
        print("="*60)
        for name, param in model.named_parameters():
            param.requires_grad = False
        layer_names = list(dict(model.named_children()).keys())
        for name, param in model.named_parameters():
            if 'layer' in name:
                parts = name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_num = int(parts[1])
                    if layer_num >= 5:
                        param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Phase 2 trainable params: {trainable:,}")
        history2, best_acc = train_model(
            model, train_loader, val_loader,
            epochs=phase2_epochs, lr=1e-5,
            label_smoothing=0.05, model_name="ResNet50_phase2",
        )
        combined_history = {k: history1[k] + history2[k] for k in history1}
        return model, combined_history, best_acc

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def main():
    print("="*60)
    print("  MODEL TRAINING — Sentinel-2 Land Type Classification")
    print("="*60)
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        batch_size=64, num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet50", choices=["custom_cnn", "resnet50"])
    parser.add_argument("--phase1_epochs", type=int, default=15)
    parser.add_argument("--phase2_epochs", type=int, default=50)
    args = parser.parse_args()
    model, history, best_acc = train_two_phase(
        train_loader, val_loader,
        architecture=args.arch,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
    )
    torch.save(model.state_dict(), f"models/{args.arch}_final.pth")
    print(f"\n[INFO] Final model saved to models/{args.arch}_final.pth")
    print(f"[RESULT] Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
