import os
import sys
import warnings
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _get_composition_aux_config(config: dict) -> dict:
    training_cfg = config.get("training", {}) if config else {}
    comp_cfg = training_cfg.get("composition_aux", {})
    return comp_cfg if comp_cfg.get("enabled", False) else {}


def _build_composition_target_table(config: dict, device: torch.device) -> torch.Tensor:
    comp_cfg = config["training"]["composition_aux"]
    target_cfg = comp_cfg["targets"]
    table = torch.tensor(
        [
            target_cfg["clay"],
            target_cfg["sand"],
            target_cfg["silt"],
        ],
        dtype=torch.float32,
        device=device,
    )
    return table


def _build_composition_targets(labels: torch.Tensor, config: dict, device: torch.device) -> torch.Tensor:
    table = _build_composition_target_table(config, device)
    return table[labels]


def _composition_reg_loss(comp_pred: torch.Tensor, comp_target: torch.Tensor, mode: str = "kl") -> torch.Tensor:
    mode = (mode or "kl").lower()
    if mode == "mse":
        return F.mse_loss(comp_pred, comp_target)
    if mode != "kl":
        raise ValueError(f"unsupported composition loss mode: {mode}")
    return F.kl_div(comp_pred.clamp_min(1e-8).log(), comp_target, reduction="batchmean")


def _composition_consistency_loss(prob_cls: torch.Tensor, comp_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prob_cls, comp_pred.detach())


def _forward_model(model: nn.Module, data: torch.Tensor, window_features: torch.Tensor, return_aux: bool = False):
    if return_aux:
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        if "return_aux" in params:
            return model(data, window_features, return_aux=True)
    return model(data, window_features)


def _unpack_model_output(model_out) -> tuple:
    if isinstance(model_out, dict):
        logits_cls = model_out["logits_cls"]
        prob_cls = model_out.get("prob_cls", torch.softmax(logits_cls, dim=1))
        comp_pred = model_out.get("comp_pred")
        return logits_cls, prob_cls, comp_pred
    logits_cls = model_out
    prob_cls = torch.softmax(logits_cls, dim=1)
    return logits_cls, prob_cls, None


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
    config: dict = None,
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
    comp_cfg = _get_composition_aux_config(config)
    use_comp_aux = bool(comp_cfg)
    warmup_epochs = int(comp_cfg.get("warmup_epochs", 0)) if use_comp_aux else 0
    use_aux_losses = use_comp_aux and epoch > warmup_epochs
    reg_loss_mode = comp_cfg.get("loss", "kl") if use_comp_aux else "kl"
    lambda_reg = float(comp_cfg.get("lambda_reg", 0.0)) if use_aux_losses else 0.0
    lambda_cons = float(comp_cfg.get("lambda_cons", 0.0)) if use_aux_losses else 0.0

    desc = f"Fold {fold} Epoch {epoch} [Train]" if fold > 0 else f"Epoch {epoch} [Train]"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_cons_loss = 0.0
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
                model_out = _forward_model(model, mixed_data, mixed_wf, return_aux=use_comp_aux)
                logits, prob_cls, comp_pred = _unpack_model_output(model_out)
                cls_loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
                if use_aux_losses and comp_pred is not None:
                    comp_target = lam * _build_composition_targets(labels, config, device) + (
                        (1 - lam) * _build_composition_targets(labels[perm], config, device)
                    )
                    reg_loss = _composition_reg_loss(comp_pred, comp_target, mode=reg_loss_mode)
                    cons_loss = _composition_consistency_loss(prob_cls, comp_pred)
                else:
                    reg_loss = logits.new_zeros(())
                    cons_loss = logits.new_zeros(())
                loss = cls_loss + lambda_reg * reg_loss + lambda_cons * cons_loss
            else:
                model_out = _forward_model(model, data, window_features, return_aux=use_comp_aux)
                logits, prob_cls, comp_pred = _unpack_model_output(model_out)
                cls_loss = criterion(logits, labels)
                if use_aux_losses and comp_pred is not None:
                    comp_target = _build_composition_targets(labels, config, device)
                    reg_loss = _composition_reg_loss(comp_pred, comp_target, mode=reg_loss_mode)
                    cons_loss = _composition_consistency_loss(prob_cls, comp_pred)
                else:
                    reg_loss = logits.new_zeros(())
                    cons_loss = logits.new_zeros(())
                loss = cls_loss + lambda_reg * reg_loss + lambda_cons * cons_loss

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
        total_cls_loss += float(cls_loss.item())
        total_reg_loss += float(reg_loss.item())
        total_cons_loss += float(cons_loss.item())
        correct += batch_correct
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            cls=f"{float(cls_loss.item()):.4f}",
            acc=f"{batch_correct / labels.size(0):.3f}",
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return {
        "train_loss": avg_loss,
        "accuracy": accuracy,
        "cls_loss": total_cls_loss / len(train_loader),
        "reg_loss": total_reg_loss / len(train_loader),
        "cons_loss": total_cons_loss / len(train_loader),
    }


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
    comp_cfg = _get_composition_aux_config(config)

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
    if comp_cfg and fold == 1:
        logger.info(
            "  Composition aux enabled | "
            f"warmup={comp_cfg.get('warmup_epochs', 0)} "
            f"lambda_reg={comp_cfg.get('lambda_reg', 0.0)} "
            f"lambda_cons={comp_cfg.get('lambda_cons', 0.0)} "
            f"loss={comp_cfg.get('loss', 'kl')}"
        )

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, fold=fold, mixup_alpha=mixup_alpha, scaler=scaler, config=config,
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
            eval_model, val_loader, val_criterion, device, class_names=class_names, config=config
        )

        if writer:
            prefix = f"fold{fold}/"
            writer.add_scalar(f"{prefix}Loss/train", train_metrics["train_loss"], epoch)
            writer.add_scalar(f"{prefix}Loss/train_cls", train_metrics["cls_loss"], epoch)
            writer.add_scalar(f"{prefix}Loss/train_reg", train_metrics["reg_loss"], epoch)
            writer.add_scalar(f"{prefix}Loss/train_cons", train_metrics["cons_loss"], epoch)
            writer.add_scalar(f"{prefix}Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar(f"{prefix}Loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar(f"{prefix}Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar(f"{prefix}F1/macro", val_metrics["macro_f1"], epoch)
            writer.add_scalar(f"{prefix}LR", optimizer.param_groups[0]["lr"], epoch)
            if val_metrics.get("mean_comp_entropy") is not None:
                writer.add_scalar(f"{prefix}Aux/mean_comp_entropy", val_metrics["mean_comp_entropy"], epoch)
            if val_metrics.get("mean_cls_comp_gap") is not None:
                writer.add_scalar(f"{prefix}Aux/mean_cls_comp_gap", val_metrics["mean_cls_comp_gap"], epoch)
            if val_metrics.get("silt_comp_mean") is not None:
                writer.add_scalar(f"{prefix}Aux/silt_comp_mean", val_metrics["silt_comp_mean"], epoch)

        logger.info(
            f"  Fold {fold} Epoch {epoch}/{total_epochs} | "
            f"训练 Loss: {train_metrics['train_loss']:.4f} "
            f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f}, cons={train_metrics['cons_loss']:.4f}) "
            f"Acc: {train_metrics['accuracy']:.4f} | "
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
