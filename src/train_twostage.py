"""
两阶段层级分类 + 多种子集成 + TTA（测试时增强）。

Stage 1: 砂土(1) vs 非砂土(0)  —— 简单任务，Cohen's d > 1.0
Stage 2: 粘土(0) vs 粉土(1)    —— 困难任务，专用模型精细区分

推理流程:
  样本 → Stage1 预测 → 如果"砂土" → 最终=砂土
                       如果"非砂土" → Stage2 预测 → 粘土 or 粉土

集成: 每阶段训练 3 个不同 seed 的模型，推理时平均 logits
TTA:  每个样本预测 1次原始 + N次增强，平均 logits
"""

import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import GammaSpectrumDataset, scan_directory
from src.model import DualBranchSEModel
from src.train import EarlyStopping, build_optimizer, build_scheduler
from src.utils import get_logger, load_config, set_seed

warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step\\(\\)`")

CLASS_NAMES_3 = ["粘土", "砂土", "粉土"]


def make_stage_config(config, num_classes):
    """复制配置并修改 num_classes。"""
    cfg = copy.deepcopy(config)
    cfg["data"]["num_classes"] = num_classes
    return cfg


# ===================================================================
#  训练单个模型
# ===================================================================
def train_stage(config, train_ds, val_ds, device, logger, tag, seed):
    """训练单阶段的一个模型，返回 (model, best_val_acc)。"""
    set_seed(seed)

    bs = config["training"]["batch_size"]
    nw = config["training"].get("num_workers", 0)

    weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    model = DualBranchSEModel(config).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"].get("label_smoothing", 0.0)
    )
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    patience = config["training"].get("early_stopping_patience", 30)
    stopper = EarlyStopping(patience) if patience > 0 else None

    best_acc = 0.0
    best_state = None
    total_epochs = config["training"]["epochs"]

    for epoch in range(1, total_epochs + 1):
        # ---- 训练 ----
        model.train()
        t_correct = 0
        t_total = 0
        for data, wf, labels in train_loader:
            data, wf, labels = data.to(device), wf.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(data, wf)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        # ---- 验证 ----
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for data, wf, labels in val_loader:
                data, wf, labels = data.to(device), wf.to(device), labels.to(device)
                logits = model(data, wf)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        scheduler.step()
        val_acc = v_correct / max(v_total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if stopper and stopper.step(val_acc):
            logger.info(
                f"  {tag} Seed{seed}: Early Stop @ Epoch {epoch}, "
                f"Best Val Acc = {best_acc:.4f}"
            )
            break
    else:
        logger.info(
            f"  {tag} Seed{seed}: 完成 {total_epochs} epochs, "
            f"Best Val Acc = {best_acc:.4f}"
        )

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


# ===================================================================
#  集成 + TTA 推理
# ===================================================================
def predict_with_tta(models, dataset, device, n_aug=10, batch_size=64):
    """
    集成 + TTA 预测。

    对每个样本:
      1. 原始输入 → 所有模型 logits 求和
      2. N次增强输入 → 所有模型 logits 求和
      3. 总 logits 取 argmax

    Returns: (preds, true_labels)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    def collect():
        all_logits = []
        all_labels = []
        for data, wf, labels in loader:
            data, wf = data.to(device), wf.to(device)
            batch_logits = None
            for m in models:
                m.eval()
                with torch.no_grad():
                    out = m(data, wf).cpu()
                if batch_logits is None:
                    batch_logits = out
                else:
                    batch_logits = batch_logits + out
            all_logits.append(batch_logits)
            all_labels.extend(labels.tolist())
        return torch.cat(all_logits, dim=0), all_labels

    orig_is_train = dataset.is_train

    # 1) 原始预测（无增强）
    dataset.is_train = False
    accumulated, labels = collect()

    # 2) N 次增强预测
    dataset.is_train = True
    for _ in range(n_aug):
        aug_logits, _ = collect()
        accumulated = accumulated + aug_logits

    dataset.is_train = orig_is_train

    preds = accumulated.argmax(dim=1).numpy()
    return preds, np.array(labels)


# ===================================================================
#  主函数
# ===================================================================
def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json",
    )
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = os.path.join(config["output"]["log_dir"], "train_twostage.log")
    logger = get_logger("twostage", log_file=log_file)
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载全部数据 ----
    all_fps, all_lbs, all_mts = [], [], []
    for d in [config["data"]["train_dir"], config["data"]["val_dir"]]:
        fps, lbs, mts, _ = scan_directory(d)
        all_fps.extend(fps)
        all_lbs.extend(lbs)
        all_mts.extend(mts)

    all_fps = np.array(all_fps)
    all_lbs = np.array(all_lbs)
    all_mts = np.array(all_mts)

    logger.info(f"数据总量: {len(all_fps)}")
    for c in range(3):
        logger.info(f"  {CLASS_NAMES_3[c]}: {(all_lbs == c).sum()}")

    # ---- 设置 ----
    n_splits = config["training"].get("n_splits", 5)
    ensemble_seeds = [42, 43, 44]
    n_tta = 10

    logger.info(
        f"策略: 两阶段层级分类 + "
        f"{len(ensemble_seeds)}-Seed 集成 + TTA(n={n_tta})"
    )
    logger.info(f"{n_splits}-Fold 交叉验证")

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=config["training"]["seed"],
    )
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_fps, all_lbs)):
        fold = fold_idx + 1
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fold {fold}/{n_splits}")

        tr_fps = all_fps[train_idx].tolist()
        tr_lbs = all_lbs[train_idx].tolist()
        tr_mts = all_mts[train_idx].tolist()
        va_fps = all_fps[val_idx].tolist()
        va_lbs = all_lbs[val_idx].tolist()
        va_mts = all_mts[val_idx].tolist()

        logger.info(f"  训练: {len(tr_fps)}  验证: {len(va_fps)}")

        # ============================================================
        #  STAGE 1: 砂土(1) vs 非砂土(0)
        # ============================================================
        logger.info(f"--- Stage 1: 砂土 vs 非砂土 ---")
        s1_cfg = make_stage_config(config, num_classes=2)
        s1_tr_labels = [1 if lb == 1 else 0 for lb in tr_lbs]
        s1_va_labels = [1 if lb == 1 else 0 for lb in va_lbs]

        # 预计算统计量（只算一次）
        s1_base_ds = GammaSpectrumDataset(
            s1_cfg, True, tr_fps, s1_tr_labels, tr_mts,
        )
        s1_stats = s1_base_ds.stats

        s1_models = []
        for seed in ensemble_seeds:
            tr_ds = GammaSpectrumDataset(
                s1_cfg, True, tr_fps, s1_tr_labels, tr_mts, s1_stats,
            )
            va_ds = GammaSpectrumDataset(
                s1_cfg, False, va_fps, s1_va_labels, va_mts, s1_stats,
            )
            model, acc = train_stage(
                s1_cfg, tr_ds, va_ds, device, logger, f"S1-F{fold}", seed,
            )
            s1_models.append(model)

        # Stage 1 集成+TTA 预测
        s1_eval_ds = GammaSpectrumDataset(
            s1_cfg, False, va_fps, s1_va_labels, va_mts, s1_stats,
        )
        s1_preds, _ = predict_with_tta(s1_models, s1_eval_ds, device, n_tta)
        s1_acc = accuracy_score(s1_va_labels, s1_preds)
        logger.info(f"  Stage1 集成+TTA: Acc = {s1_acc:.4f}")

        # ============================================================
        #  STAGE 2: 粘土(0) vs 粉土(1)
        # ============================================================
        logger.info(f"--- Stage 2: 粘土 vs 粉土 ---")
        s2_cfg = make_stage_config(config, num_classes=2)

        # 过滤：仅保留粘土和粉土
        s2_tr_idx = [i for i, lb in enumerate(tr_lbs) if lb != 1]
        s2_tr_fps = [tr_fps[i] for i in s2_tr_idx]
        s2_tr_labels = [0 if tr_lbs[i] == 0 else 1 for i in s2_tr_idx]
        s2_tr_mts = [tr_mts[i] for i in s2_tr_idx]

        s2_va_idx = [i for i, lb in enumerate(va_lbs) if lb != 1]
        s2_va_fps = [va_fps[i] for i in s2_va_idx]
        s2_va_labels = [0 if va_lbs[i] == 0 else 1 for i in s2_va_idx]
        s2_va_mts = [va_mts[i] for i in s2_va_idx]

        logger.info(
            f"  Stage2 训练: {len(s2_tr_fps)} "
            f"(粘土={s2_tr_labels.count(0)}, 粉土={s2_tr_labels.count(1)})"
        )

        # 预计算统计量
        s2_base_ds = GammaSpectrumDataset(
            s2_cfg, True, s2_tr_fps, s2_tr_labels, s2_tr_mts,
        )
        s2_stats = s2_base_ds.stats

        s2_models = []
        for seed in ensemble_seeds:
            tr_ds = GammaSpectrumDataset(
                s2_cfg, True, s2_tr_fps, s2_tr_labels, s2_tr_mts, s2_stats,
            )
            va_ds = GammaSpectrumDataset(
                s2_cfg, False, s2_va_fps, s2_va_labels, s2_va_mts, s2_stats,
            )
            model, acc = train_stage(
                s2_cfg, tr_ds, va_ds, device, logger, f"S2-F{fold}", seed,
            )
            s2_models.append(model)

        # Stage 2 集成+TTA (仅粘土+粉土样本)
        s2_sub_ds = GammaSpectrumDataset(
            s2_cfg, False, s2_va_fps, s2_va_labels, s2_va_mts, s2_stats,
        )
        s2_sub_preds, _ = predict_with_tta(s2_models, s2_sub_ds, device, n_tta)
        s2_sub_acc = accuracy_score(s2_va_labels, s2_sub_preds)
        logger.info(f"  Stage2 集成+TTA (粘土vs粉土): Acc = {s2_sub_acc:.4f}")

        # Stage 2 预测全部验证集（用于组合评估）
        s2_all_ds = GammaSpectrumDataset(
            s2_cfg, False, va_fps, [0] * len(va_fps), va_mts, s2_stats,
        )
        s2_all_preds, _ = predict_with_tta(s2_models, s2_all_ds, device, n_tta)

        # ============================================================
        #  组合评估
        # ============================================================
        final = np.zeros(len(va_lbs), dtype=int)
        for i in range(len(va_lbs)):
            if s1_preds[i] == 1:
                final[i] = 1  # 砂土
            else:
                final[i] = 0 if s2_all_preds[i] == 0 else 2  # 粘土 or 粉土

        true = np.array(va_lbs)
        acc = accuracy_score(true, final)
        mf1 = f1_score(true, final, average="macro", zero_division=0)
        cm = confusion_matrix(true, final, labels=[0, 1, 2])

        logger.info(f"\n  Fold {fold} 组合结果: Acc={acc:.4f}  F1={mf1:.4f}")
        logger.info(
            f"\n{classification_report(true, final, target_names=CLASS_NAMES_3, zero_division=0)}"
        )
        logger.info(f"  混淆矩阵:\n{cm}")

        fold_results.append({"accuracy": acc, "macro_f1": mf1})

        # 释放 GPU 显存
        del s1_models, s2_models
        torch.cuda.empty_cache()

    # ================================================================
    #  汇总
    # ================================================================
    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    avg_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"两阶段 + {len(ensemble_seeds)}-Seed集成 + TTA({n_tta}) 最终结果"
    )
    logger.info(f"{'=' * 60}")
    for i, r in enumerate(fold_results):
        logger.info(
            f"  Fold {i + 1}: Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}"
        )
    logger.info(f"  平均 Acc: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  平均 F1:  {avg_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"{'=' * 60}")
    logger.info("训练完成。")


if __name__ == "__main__":
    main()
