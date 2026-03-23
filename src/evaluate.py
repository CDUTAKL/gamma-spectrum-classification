import os
import inspect

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DEFAULT_CLASS_NAMES = ["粘土", "砂土", "粉土"]
PLOT_NAME_FALLBACKS = {
    "粘土": "Clay",
    "砂土": "Sand",
    "粉土": "Silt",
}


def _configure_plot_text(class_names: list):
    """Pick a usable font and fallback labels for headless servers."""
    from matplotlib import font_manager

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return list(class_names), "预测标签", "真实标签"

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    safe_names = [PLOT_NAME_FALLBACKS.get(name, name) for name in class_names]
    return safe_names, "Predicted", "True"


def evaluate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list = None,
    config: dict = None,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    non_blocking = device.type == "cuda"
    comp_cfg = config.get("training", {}).get("composition_aux", {}) if config else {}
    use_comp_aux = bool(comp_cfg.get("enabled", False))
    total_comp_entropy = 0.0
    total_cls_comp_gap = 0.0
    total_silt_comp = 0.0
    total_samples = 0
    total_silt_samples = 0

    with torch.inference_mode():
        for data, window_features, labels in val_loader:
            data = data.to(device, non_blocking=non_blocking)
            window_features = window_features.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if use_comp_aux:
                try:
                    params = inspect.signature(model.forward).parameters
                except (TypeError, ValueError):
                    params = {}
                if "return_aux" in params:
                    model_out = model(data, window_features, return_aux=True)
                else:
                    model_out = model(data, window_features)
            else:
                model_out = model(data, window_features)

            if isinstance(model_out, dict):
                logits = model_out["logits_cls"]
                prob_cls = model_out.get("prob_cls", F.softmax(logits, dim=1))
                comp_pred = model_out.get("comp_pred")
            else:
                logits = model_out
                prob_cls = F.softmax(logits, dim=1)
                comp_pred = None

            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if comp_pred is not None:
                comp_safe = comp_pred.clamp_min(1e-8)
                entropy = -(comp_safe * comp_safe.log()).sum(dim=1)
                cls_comp_gap = torch.abs(prob_cls - comp_pred).mean(dim=1)
                total_comp_entropy += float(entropy.sum().item())
                total_cls_comp_gap += float(cls_comp_gap.sum().item())
                total_samples += labels.size(0)

                silt_mask = labels == 2
                if silt_mask.any():
                    total_silt_comp += float(comp_pred[silt_mask, 2].sum().item())
                    total_silt_samples += int(silt_mask.sum().item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = class_names or DEFAULT_CLASS_NAMES
    labels_range = list(range(len(class_names)))

    accuracy = (all_preds == all_labels).mean()
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=labels_range, zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_preds, average="macro", labels=labels_range, zero_division=0
    )
    weighted_f1 = f1_score(
        all_labels, all_preds, average="weighted", labels=labels_range, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)

    return {
        "val_loss": total_loss / max(len(val_loader), 1),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "class_names": class_names,
        "mean_comp_entropy": (
            total_comp_entropy / total_samples if total_samples > 0 else None
        ),
        "mean_cls_comp_gap": (
            total_cls_comp_gap / total_samples if total_samples > 0 else None
        ),
        "silt_comp_mean": (
            total_silt_comp / total_silt_samples if total_silt_samples > 0 else None
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    epoch: int,
    save_path: str = None,
) -> plt.Figure:
    display_names, xlabel, ylabel = _configure_plot_text(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    class_names: list = None,
) -> None:
    class_names = class_names or metrics.get("class_names") or DEFAULT_CLASS_NAMES
    writer.add_scalar("Loss/val", metrics["val_loss"], epoch)
    writer.add_scalar("Accuracy/val", metrics["accuracy"], epoch)
    writer.add_scalar("F1/macro", metrics["macro_f1"], epoch)
    writer.add_scalar("F1/weighted", metrics["weighted_f1"], epoch)
    if metrics.get("mean_comp_entropy") is not None:
        writer.add_scalar("Aux/mean_comp_entropy", metrics["mean_comp_entropy"], epoch)
    if metrics.get("mean_cls_comp_gap") is not None:
        writer.add_scalar("Aux/mean_cls_comp_gap", metrics["mean_cls_comp_gap"], epoch)
    if metrics.get("silt_comp_mean") is not None:
        writer.add_scalar("Aux/silt_comp_mean", metrics["silt_comp_mean"], epoch)
    for i, name in enumerate(class_names):
        writer.add_scalar(f"F1_per_class/{name}", float(metrics["per_class_f1"][i]), epoch)

    cm_save_path = os.path.join(log_dir, f"confusion_matrix_epoch{epoch:03d}.png")
    fig = plot_confusion_matrix(
        metrics["confusion_matrix"], class_names, epoch, save_path=cm_save_path
    )
    writer.add_figure("Confusion_Matrix", fig, epoch)
    plt.close(fig)
