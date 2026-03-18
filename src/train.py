import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step\\(\\)`")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_kfold_dataloaders
from src.evaluate import evaluate_epoch
from src.model import DualBranchSEModel, FocalLoss
from src.utils import CheckpointManager, get_logger, load_config, set_seed


def build_optimizer(model: nn.Module, config: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def build_scheduler(optimizer, config: dict):
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["epochs"]
    base_lr = config["training"]["learning_rate"]

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=base_lr * 0.01,
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    return cosine_scheduler


def build_criterion(config: dict, device: torch.device):
    """构建损失函数：Focal Loss 或 CrossEntropy。"""
    loss_type = config["training"].get("loss_type", "focal")
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    num_classes = config["data"]["num_classes"]

    if loss_type == "focal":
        gamma = config["training"].get("focal_gamma", 2.0)
        # 类别平衡权重：粉土样本最少，权重最大
        alpha = config["training"].get("focal_alpha", None)
        if alpha is not None:
            alpha = torch.FloatTensor(alpha).to(device)
        return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    fold: int = 0,
    mixup_alpha: float = 0.0,
    scaler=None,
) -> dict:
    """训练一个 epoch。

    Args:
        scaler: AMP GradScaler 实例。传入时启用混合精度训练,
                前向/反向用 FP16 加速, 权重更新保持 FP32 精度。
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    non_blocking = device.type == "cuda"

    desc = f"Fold {fold} Epoch {epoch} [Train]" if fold > 0 else f"Epoch {epoch} [Train]"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for batch_idx, (data, window_features, labels) in enumerate(pbar):
        data = data.to(device, non_blocking=non_blocking)
        window_features = window_features.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        # AMP: autocast 块内的运算自动选择 FP16/FP32
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Mixup 数据增强
            if mixup_alpha > 0 and model.training:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(data.size(0), device=device)
                mixed_data = lam * data + (1 - lam) * data[perm]
                mixed_wf = lam * window_features + (1 - lam) * window_features[perm]
                logits = model(mixed_data, mixed_wf)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
            else:
                logits = model(data, window_features)
                loss = criterion(logits, labels)

        # AMP: scaler 处理反向传播和权重更新
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_loss = loss.item()
        preds = logits.argmax(dim=1)
        batch_correct = (preds == labels).sum().item()

        total_loss += batch_loss
        correct += batch_correct
        total += labels.size(0)

        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_correct / labels.size(0):.3f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return {"train_loss": avg_loss, "accuracy": accuracy}


class EarlyStopping:
    def __init__(self, patience: int = 30, min_epochs: int = 0):
        self.patience = patience
        self.min_epochs = max(int(min_epochs), 0)
        self.counter = 0
        self.best_score = -float("inf")

    def step(self, score: float, epoch=None) -> bool:
        """更新 early stopping 计数；返回 True 表示应当停止。

        在 Python 3.9 环境下不使用 ``int | None`` 这类联合类型写法，
        保持与 5090/本机两侧解释器兼容。
        """
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False

        # 在达到最小训练轮数前，不累计 early stopping 计数
        if epoch is not None and epoch < self.min_epochs:
            return False

        self.counter += 1
        return self.counter >= self.patience


def train_single_fold(
    config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    logger,
    fold: int,
    writer: SummaryWriter = None,
) -> dict:
    model = DualBranchSEModel(config).to(device)
    if fold == 1:
        logger.info(f"  模型参数量: {model.get_num_params():,}")

    # torch.compile: 图编译优化, 自动融合算子 (PyTorch 2.x)
    if config["training"].get("use_compile", False) and device.type == "cuda":
        model = torch.compile(model, backend="aot_eager")
        if fold == 1:
            logger.info("  已启用 torch.compile(backend='aot_eager') 图编译")

    criterion = build_criterion(config, device)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    patience = config["training"].get("early_stopping_patience", 0)
    min_epochs = config["training"].get("min_epochs_before_early_stop", 0)
    early_stopper = (
        EarlyStopping(patience=patience, min_epochs=min_epochs)
        if patience > 0 else None
    )

    ckpt_dir = config["output"]["checkpoint_dir"]
    ckpt_manager = CheckpointManager(ckpt_dir)

    total_epochs = config["training"]["epochs"]
    mixup_alpha = config.get("augmentation", {}).get("mixup_alpha", 0.0)
    best_metrics = None

    # AMP 混合精度: CUDA 设备上启用, 加速 15-25%
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # SWA 配置
    swa_start_frac = config["training"].get("swa_start_epoch", 0)
    swa_start = int(total_epochs * swa_start_frac) if 0 < swa_start_frac < 1 else int(swa_start_frac)
    use_swa = swa_start > 0
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr = config["training"].get("swa_lr", 1e-4)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    # 验证频率: 每 val_interval 个 epoch 验证一次, 减少验证开销
    val_interval = config["training"].get("val_interval", 1)
    warmup_epochs = config["training"].get("warmup_epochs", 5)

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, fold=fold, mixup_alpha=mixup_alpha, scaler=scaler,
        )

        # SWA 阶段：更新平均模型，使用 SWA 学习率
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # 跳过非验证 epoch: 节省验证时间
        need_val = (
            epoch % val_interval == 0       # 每隔 val_interval 验证
            or epoch == total_epochs         # 最后一个 epoch 必须验证
            or epoch == warmup_epochs + 1    # warmup 刚结束时验证一次
        )
        if not need_val:
            continue

        # 验证：SWA 阶段后用 swa_model 做评估
        eval_model = model
        if use_swa and epoch >= swa_start and swa_model is not None:
            # 更新 BN 统计量
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            eval_model = swa_model

        class_names = config.get("data", {}).get("class_names")
        val_metrics = evaluate_epoch(
            eval_model, val_loader, val_criterion, device, class_names=class_names
        )

        if writer:
            prefix = f"fold{fold}/"
            writer.add_scalar(f"{prefix}Loss/train", train_metrics["train_loss"], epoch)
            writer.add_scalar(f"{prefix}Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar(f"{prefix}Loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar(f"{prefix}Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar(f"{prefix}F1/macro", val_metrics["macro_f1"], epoch)
            writer.add_scalar(f"{prefix}LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"  Fold {fold} Epoch {epoch}/{total_epochs} | "
            f"训练 Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"验证 Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['macro_f1']:.4f}"
            + (" [SWA]" if use_swa and epoch >= swa_start else "")
        )

        is_best = val_metrics["accuracy"] > ckpt_manager.best_val_acc
        if is_best:
            best_metrics = val_metrics.copy()
            ckpt_manager.save(model, optimizer, epoch, val_metrics, is_best=True)
            logger.info(f"  Fold {fold} 新最优！验证 Acc: {val_metrics['accuracy']:.4f}")

        if early_stopper and early_stopper.step(val_metrics["accuracy"], epoch=epoch):
            logger.info(
                f"  Fold {fold} Early Stopping @ Epoch {epoch}，"
                f"最佳 Acc: {early_stopper.best_score:.4f}"
            )
            break

    return best_metrics


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json"
    )
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = os.path.join(config["output"]["log_dir"], "train.log")
    logger = get_logger("train", log_file=log_file)
    logger.info(f"使用设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    n_splits = config["training"].get("n_splits", 5)
    loss_type = config["training"].get("loss_type", "focal")
    logger.info(f"=== {n_splits}-Fold 交叉验证 | 损失函数: {loss_type} ===")

    folds = build_kfold_dataloaders(config, n_splits=n_splits)

    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["output"]["log_dir"])

    fold_results = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        fold = fold_idx + 1
        logger.info(f"\n--- Fold {fold}/{n_splits} ---")
        logger.info(f"  训练: {len(train_loader.dataset)}  验证: {len(val_loader.dataset)}")

        ckpt_dir = config["output"]["checkpoint_dir"]
        for f_name in ["best_model.pth", "last_model.pth"]:
            ckpt_path = os.path.join(ckpt_dir, f_name)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        best_metrics = train_single_fold(
            config, train_loader, val_loader, device, logger, fold, writer
        )

        if best_metrics:
            fold_results.append(best_metrics)
            logger.info(
                f"  Fold {fold} 最佳: Acc={best_metrics['accuracy']:.4f} "
                f"F1={best_metrics['macro_f1']:.4f}"
            )

    writer.close()

    if fold_results:
        avg_acc = np.mean([r["accuracy"] for r in fold_results])
        std_acc = np.std([r["accuracy"] for r in fold_results])
        avg_f1 = np.mean([r["macro_f1"] for r in fold_results])
        std_f1 = np.std([r["macro_f1"] for r in fold_results])

        logger.info(f"\n{'=' * 50}")
        logger.info(f"=== {n_splits}-Fold 交叉验证结果 ===")
        logger.info(f"{'=' * 50}")
        for i, r in enumerate(fold_results):
            logger.info(f"  Fold {i + 1}: Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}")
        logger.info(f"  平均 Acc: {avg_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"  平均 F1:  {avg_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"{'=' * 50}")

    logger.info("训练完成。")


if __name__ == "__main__":
    main()
