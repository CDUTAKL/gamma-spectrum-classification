import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CLASS_NAMES = ["粘土", "砂土", "粉土"]


def evaluate_epoch(
        
    model: nn.Module,

    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for data, window_features, labels in val_loader:
            data = data.to(device)
            window_features = window_features.to(device)
            labels = labels.to(device)
            logits = model(data, window_features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels_range = list(range(len(CLASS_NAMES)))

    accuracy = (all_preds == all_labels).mean()
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                            labels=labels_range, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        labels=labels_range, zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted",
                           labels=labels_range, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)

    return {
        "val_loss": total_loss / max(len(val_loader), 1),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    epoch: int,
    save_path: str = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=100)
    return fig


def log_to_tensorboard(
    writer: SummaryWriter,
    metrics: dict,
    epoch: int,
    log_dir: str,
) -> None:
    writer.add_scalar("Loss/val", metrics["val_loss"], epoch)
    writer.add_scalar("Accuracy/val", metrics["accuracy"], epoch)
    writer.add_scalar("F1/macro", metrics["macro_f1"], epoch)
    writer.add_scalar("F1/weighted", metrics["weighted_f1"], epoch)
    for i, name in enumerate(CLASS_NAMES):
        writer.add_scalar(f"F1_per_class/{name}",
                          float(metrics["per_class_f1"][i]), epoch)

    cm_save_path = os.path.join(log_dir, f"confusion_matrix_epoch{epoch:03d}.png")
    fig = plot_confusion_matrix(
        metrics["confusion_matrix"], CLASS_NAMES, epoch, save_path=cm_save_path
    )
    writer.add_figure("Confusion_Matrix", fig, epoch)
    plt.close(fig)
