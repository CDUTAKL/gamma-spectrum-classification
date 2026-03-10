"""
三分支异构集成训练系统 v2

=== 五层架构创新 ===

  1. TriBranch 模型
     Multi-Scale SE-CNN + Spectral Transformer + MLP 三分支融合
     - CNN 捕获局部峰特征, Transformer 捕获全局谱段关联, MLP 编码物理先验

  2. 异构集成
     TriBranch CNN x 3 (不同 seed) + GradientBoosting + XGBoost
     - 深度学习 + 传统 ML 互补, 降低单一模型偏差

  3. Stacking 元学习器
     XGBoost 元学习器在 out-of-fold 预测上训练
     - 自适应学习每个基模型在不同类别上的可靠性
     - 比固定权重融合提升 1-3% (文献: ScienceDirect, 2023)

  4. TTA 测试时增强
     原始 + 10 次增强版本的 logits 平均
     - 降低单次预测的随机性, 稳定边界样本

  5. Gradient Centralization
     梯度零均值中心化正则 (arxiv:2004.01461)
     - 对 <3000 样本的小数据集有显著正则化效果

  6. SMOTE 特征空间过采样
     对 ML 基分类器的工程特征做 SMOTE, 增强粉土少数类
     - 在特征空间插值比原始空间更合理 (ISMOTE, Nature 2025)

=== 执行流程 ===

  Phase 1 — 5-Fold 收集 out-of-fold 预测
    for each fold:
      a) 训练 TriBranch x 3 (不同 seed) -> TTA 预测 -> oof_cnn
      b) 训练 GB (SMOTE) -> 预测 -> oof_gb
      c) 训练 XGBoost (SMOTE) -> 预测 -> oof_xgb

  Phase 2 — Stacking 元学习器评估
    meta_X = concat(oof_cnn, oof_gb, oof_xgb)  # (N, 9)
    XGBoost 元学习器 5-Fold CV -> 最终结果

用法: python src/train_ensemble.py
"""

import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from xgboost import XGBClassifier

warnings.filterwarnings(
    "ignore",
    message="The epoch parameter in `scheduler.step\\(\\)`",
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.dataset import (
    GammaSpectrumDataset,
    extract_engineered_features,
    load_spectrum,
    scan_directory,
)
from src.model import TriBranchModel
from src.train import (
    EarlyStopping,
    build_criterion,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
)
from src.evaluate import evaluate_epoch
from src.utils import get_logger, load_config, set_seed
from src.artifacts import save_stacking_oof_artifacts

DEFAULT_CLASS_NAMES = ["粘土", "砂土", "粉土"]


# =====================================================================
#  工具函数
# =====================================================================

def apply_gradient_centralization(model: nn.Module) -> None:
    """对模型所有多维参数注册梯度中心化钩子。

    Gradient Centralization (Yong et al., 2020):
      对每个权重矩阵的梯度减去行均值, 约束梯度方向到超平面上,
      起到隐式正则化作用, 对小数据集尤其有效。

    钩子在 loss.backward() 时自动触发, 无需修改训练循环。
    """
    def _gc_hook(grad):
        if len(grad.shape) > 1:
            return grad - grad.mean(
                dim=tuple(range(1, len(grad.shape))), keepdim=True
            )
        return None  # 1D 参数 (bias, BN) 不做中心化

    for name, p in model.named_parameters():
        if p.requires_grad and len(p.shape) > 1:
            p.register_hook(_gc_hook)


def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k: int = 5) -> tuple:
    """在特征空间中对少数类执行 SMOTE 过采样。

    对每个少数类生成合成样本, 使所有类别数量与最多类持平。
    合成样本 = 原样本 + lambda * (近邻样本 - 原样本), lambda ~ U(0,1)

    Args:
        X: (N, D) 特征矩阵
        y: (N,) 标签
        k: 近邻数

    Returns:
        (X_resampled, y_resampled)
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_parts = [X]
    y_parts = [y]

    for cls, cnt in zip(classes, counts):
        n_synthetic = max_count - cnt
        if n_synthetic <= 0:
            continue

        # 提取当前类的样本
        mask = y == cls
        X_cls = X[mask]
        n_cls = len(X_cls)

        # 拟合近邻模型
        nn_model = NearestNeighbors(
            n_neighbors=min(k + 1, n_cls), metric='euclidean'
        )
        nn_model.fit(X_cls)
        nn_indices = nn_model.kneighbors(
            X_cls, return_distance=False
        )[:, 1:]  # 排除自身

        # 生成合成样本
        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=X.dtype)
        for i in range(n_synthetic):
            # 随机选一个原始样本
            idx = np.random.randint(n_cls)
            # 随机选一个近邻
            nn_idx = nn_indices[
                idx, np.random.randint(nn_indices.shape[1])
            ]
            # 在两点之间线性插值
            lam = np.random.rand()
            synthetic[i] = X_cls[idx] + lam * (
                X_cls[nn_idx] - X_cls[idx]
            )

        X_parts.append(synthetic)
        y_parts.append(np.full(n_synthetic, cls, dtype=y.dtype))

    return np.vstack(X_parts), np.concatenate(y_parts)


# =====================================================================
#  Phase 1: 基模型训练与预测
# =====================================================================

def train_cnn_model(config, train_ds, val_ds, device, logger, tag, seed):
    """训练单个 TriBranch CNN 模型。

    流程:
      1. 设置随机种子 -> 确保不同 seed 的模型有不同初始化
      2. 创建 TriBranchModel + Gradient Centralization 钩子
      3. Focal Loss + AdamW + Cosine Warmup + AMP 混合精度训练
      4. Early Stopping 选择最优模型

    AMP 混合精度:
      前向/反向用 FP16 加速计算, 权重更新保持 FP32 精度。
      GradScaler 防止 FP16 梯度下溢, 自动调整缩放因子。

    Args:
        config:   完整配置字典
        train_ds: 训练集 GammaSpectrumDataset
        val_ds:   验证集 GammaSpectrumDataset
        device:   cuda / cpu
        logger:   日志器
        tag:      日志标签 (e.g. "F1")
        seed:     随机种子

    Returns:
        (model, best_val_acc)
    """
    set_seed(seed)

    bs = config["training"]["batch_size"]
    nw = config["training"].get("num_workers", 0)
    mixup_alpha = config.get("augmentation", {}).get("mixup_alpha", 0.0)

    # 类别加权采样: 平衡粘土/砂土/粉土的采样频率
    weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    # 模型初始化 + Gradient Centralization
    model = TriBranchModel(config).to(device)
    apply_gradient_centralization(model)

    # ---- 加载 SMAE 预训练权重 (若存在) ----
    smae_path = os.path.join(
        config["output"]["checkpoint_dir"], "smae_pretrained.pth"
    )
    if os.path.exists(smae_path):
        ckpt = torch.load(smae_path, map_location=device, weights_only=True)
        encoder_state = ckpt["encoder_state_dict"]
        # 将 SMAE 编码器权重加载到 transformer_branch
        # 加前缀 "transformer_branch." 适配 TriBranchModel 的 state_dict
        prefixed = {
            f"transformer_branch.{k}": v for k, v in encoder_state.items()
        }
        missing, unexpected = model.load_state_dict(prefixed, strict=False)
        if seed == 42:
            logger.info(
                f"  SMAE 预训练权重已加载 "
                f"(Loss={ckpt.get('best_loss', '?'):.6f}, "
                f"迁移 {len(prefixed)} 个张量)"
            )
    else:
        if seed == 42:
            logger.info("  未找到 SMAE 预训练权重, 使用随机初始化")

    if seed == 42:
        logger.info(f"  模型参数量: {model.get_num_params():,}")

    # torch.compile: 图编译优化, 自动融合算子 (PyTorch 2.x)
    # 输入尺寸固定 (820), 非常适合编译加速, 预计提速 15-30%
    if config["training"].get("use_compile", False) and device.type == "cuda":
        model = torch.compile(model, backend="aot_eager")
        if seed == 42:
            logger.info("  已启用 torch.compile(backend='aot_eager') 图编译")

    criterion = build_criterion(config, device)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # AMP: 仅在 CUDA 设备上启用混合精度
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    patience = config["training"].get("early_stopping_patience", 30)
    stopper = EarlyStopping(patience) if patience > 0 else None

    best_acc = 0.0
    best_state = None
    total_epochs = config["training"]["epochs"]

    # 验证频率: 每 val_interval 个 epoch 验证一次, 减少验证开销
    # 最后一个 epoch 和 warmup 结束后必须验证, 确保不遗漏最优模型
    val_interval = config["training"].get("val_interval", 1)
    warmup_epochs = config["training"].get("warmup_epochs", 10)

    for epoch in range(1, total_epochs + 1):
        # 训练一个 epoch (传入 scaler 启用 AMP)
        train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, mixup_alpha=mixup_alpha, scaler=scaler,
        )

        scheduler.step()

        # 跳过非验证 epoch: 节省验证时间
        need_val = (
            epoch % val_interval == 0       # 每隔 val_interval 验证
            or epoch == total_epochs         # 最后一个 epoch 必须验证
            or epoch == warmup_epochs + 1    # warmup 刚结束时验证一次
        )
        if not need_val:
            continue

        # 验证
        val_metrics = evaluate_epoch(
            model,
            val_loader,
            val_criterion,
            device,
            class_names=config.get("data", {}).get("class_names"),
        )
        val_acc = val_metrics["accuracy"]

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if stopper and stopper.step(val_acc):
            logger.info(
                f"  {tag} Seed{seed}: "
                f"Early Stop @ Epoch {epoch}, "
                f"Best Val Acc = {best_acc:.4f}"
            )
            break
    else:
        logger.info(
            f"  {tag} Seed{seed}: "
            f"完成 {total_epochs} epochs, "
            f"Best Val Acc = {best_acc:.4f}"
        )

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def predict_cnn_tta(models, dataset, device,
                    n_tta=3, batch_size=64,
                    clean_weight=0.7):
    """CNN 集成 + TTA 预测概率。

    改进的 TTA 策略 (v2):
      - 仅使用 Poisson 重采样 (物理上合理, 不破坏峰位置)
      - 减少增强次数 (10 -> 3), 避免过度噪声
      - 加权平均: 干净预测权重 70%, 增强预测权重 30%
        原因: 干净输入最可靠, 增强仅提供额外的统计多样性

    推理流程:
      1. 原始输入 (tta_mode=False): 所有模型 logits 平均 -> clean_logits
      2. N 次 Poisson 重采样 (tta_mode=True): 同上 -> 平均 -> aug_logits
      3. final = clean_weight * clean + (1-clean_weight) * aug -> softmax

    Args:
        models:       训练好的模型列表
        dataset:      GammaSpectrumDataset
        device:       cuda / cpu
        n_tta:        TTA Poisson 重采样次数 (默认 3)
        batch_size:   推理批大小
        clean_weight: 干净预测的权重 (默认 0.7)

    Returns:
        (probs, true_labels): probs shape = (N, num_classes)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    def _collect_logits():
        """一次前向传播, 返回所有样本的集成平均 logits。"""
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
            # 模型间平均
            all_logits.append(batch_logits / len(models))
            all_labels.extend(labels.tolist())
        return torch.cat(all_logits, dim=0), np.array(all_labels)

    # 保存原始状态
    orig_is_train = dataset.is_train
    orig_tta_mode = dataset.tta_mode

    # 第 1 步: 干净输入 (无任何增强)
    dataset.is_train = False
    dataset.tta_mode = False
    clean_logits, labels = _collect_logits()

    # 第 2 步: N 次 Poisson 重采样增强
    dataset.is_train = False
    dataset.tta_mode = True  # 仅 Poisson 重采样
    aug_accumulated = None
    for _ in range(n_tta):
        aug_logits, _ = _collect_logits()
        if aug_accumulated is None:
            aug_accumulated = aug_logits
        else:
            aug_accumulated = aug_accumulated + aug_logits
    aug_avg = aug_accumulated / n_tta

    # 第 3 步: 加权融合
    final_logits = clean_weight * clean_logits + (1 - clean_weight) * aug_avg

    # 恢复原始状态
    dataset.is_train = orig_is_train
    dataset.tta_mode = orig_tta_mode

    probs = torch.softmax(final_logits, dim=1).numpy()
    return probs, labels


def _meta_uncertainty_features(probs: np.ndarray) -> np.ndarray:
    """Uncertainty features from a (N, C) probability matrix.

    Returns (N, 3): [max_prob, margin_top1_top2, entropy]
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {p.shape}")
    top1 = p.max(axis=1)
    sort2 = np.sort(p, axis=1)
    top2 = (
        sort2[:, -2]
        if sort2.shape[1] >= 2
        else np.zeros(p.shape[0], dtype=np.float64)
    )
    margin = top1 - top2
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    return np.stack([top1, margin, entropy], axis=1).astype(np.float32)


def build_meta_features(
    oof_cnn: np.ndarray,
    oof_gb: np.ndarray,
    oof_xgb: np.ndarray,
    measure_times: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build meta features for stacking.

    mode:
      - "proba_only": concat [cnn, gb, xgb] -> (N, 9)
      - "proba+uncertainty": add 3*3 uncertainty features -> (N, 18)
      - "proba+uncertainty+time": add log(measure_time) -> (N, 19)
    """
    base = np.hstack([oof_cnn, oof_gb, oof_xgb]).astype(np.float32)
    mode = (mode or "proba_only").strip().lower()
    if mode == "proba_only":
        return base

    parts = [base]
    if "uncertainty" in mode:
        parts.append(_meta_uncertainty_features(oof_cnn))
        parts.append(_meta_uncertainty_features(oof_gb))
        parts.append(_meta_uncertainty_features(oof_xgb))
    if mode.endswith("+time"):
        mt = np.asarray(measure_times, dtype=np.float32)
        parts.append(np.log(np.maximum(mt, 1e-6)).reshape(-1, 1))
    return np.hstack(parts).astype(np.float32)


def extract_ml_features(file_paths, measure_times,
                        energy_windows, spectrum_length,
                        feature_stats=None):
    """批量提取 ML 基分类器的能窗工程特征。

    Args:
        file_paths:      文件路径列表
        measure_times:   测量时长列表
        energy_windows:  能窗配置 {"K": [lo,hi], "U": ..., "Th": ...}
        spectrum_length: 谱长度 (820)
        feature_stats:   训练集统计量；若包含 PCA 参数则追加 PCA 得分

    Returns:
        (N, D) 特征矩阵，当前 D=86
    """
    X = []
    for fp, mt in zip(file_paths, measure_times):
        raw = load_spectrum(fp, spectrum_length)
        cps = raw / mt
        features = extract_engineered_features(
            cps, energy_windows, mt, feature_stats
        )
        X.append(features)
    return np.array(X, dtype=np.float32)


# =====================================================================
#  主函数
# =====================================================================

def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json",
    )
    config = load_config(config_path)
    class_names = config.get("data", {}).get("class_names") or DEFAULT_CLASS_NAMES
    if len(class_names) != config["data"]["num_classes"]:
        raise ValueError(
            f"class_names 长度({len(class_names)})与 num_classes({config['data']['num_classes']})不一致"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = os.path.join(
        config["output"]["log_dir"], "train_ensemble_v2.log"
    )
    logger = get_logger("ensemble_v2", log_file=log_file)
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
    N = len(all_fps)
    C = config["data"]["num_classes"]

    logger.info(f"数据总量: {N}")
    for c in range(C):
        logger.info(f"  {class_names[c]}: {(all_lbs == c).sum()}")

    # ---- 超参数 ----
    n_splits = config["training"].get("n_splits", 5)
    ensemble_seeds = [42, 43, 44]     # 3-seed CNN 集成
    n_tta = 3                         # TTA Poisson 重采样次数 (从 10 降到 3)
    energy_windows = config["data"]["energy_windows"]
    spectrum_length = config["data"]["spectrum_length"]

    logger.info(f"策略: TriBranch x {len(ensemble_seeds)} + GB + XGB "
                f"+ Stacking + TTA({n_tta})")
    logger.info(f"{n_splits}-Fold 交叉验证")

    # ---- 时间分层 K-Fold ----
    # 按 (label, time_group) 联合分层, 确保每个 fold 时长分布均匀
    time_bins = np.array([0 if t <= 60 else 1 for t in all_mts])
    stratify_key = all_lbs * 10 + time_bins

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=config["training"]["seed"],
    )

    # ================================================================
    #  Phase 1: 收集 out-of-fold 预测
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 1: 训练基模型, 收集 out-of-fold 预测")
    logger.info(f"{'=' * 60}")

    # 预分配 OOF 概率矩阵 (所有样本的 out-of-fold 预测)
    oof_cnn = np.zeros((N, C), dtype=np.float64)   # CNN 集成+TTA
    oof_gb = np.zeros((N, C), dtype=np.float64)     # GradientBoosting
    oof_xgb = np.zeros((N, C), dtype=np.float64)    # XGBoost
    oof_true = np.zeros(N, dtype=np.int64)           # 真实标签

    # 逐 fold 记录各基模型的独立准确率 (用于对比分析)
    fold_cnn_acc = []
    fold_gb_acc = []
    fold_xgb_acc = []
    fold_fixed_acc = []  # 固定权重基线

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_fps, stratify_key)
    ):
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
        #  1a. CNN 集成 (TriBranch x 3 seeds)
        # ============================================================
        logger.info("--- TriBranch CNN 集成 ---")

        # 预计算统计量 (只算一次, 所有 seed 共享)
        # 注意: 统计量计算本身会完整遍历训练集；此处强制禁用 cache，避免额外的 L0 预加载造成重复 I/O。
        stats_cfg = copy.deepcopy(config)
        stats_cache = dict(stats_cfg.get("cache", {}))
        stats_cache["enabled"] = False
        stats_cfg["cache"] = stats_cache
        base_ds = GammaSpectrumDataset(stats_cfg, True, tr_fps, tr_lbs, tr_mts)
        stats = base_ds.stats
        del base_ds

        # 复用数据集对象，避免每个 seed 重建缓存（尤其是验证集 L1 缓存）。
        tr_ds = GammaSpectrumDataset(config, True, tr_fps, tr_lbs, tr_mts, stats)
        va_ds = GammaSpectrumDataset(config, False, va_fps, va_lbs, va_mts, stats)

        cnn_models = []
        for seed in ensemble_seeds:
            model, acc = train_cnn_model(
                config, tr_ds, va_ds, device, logger,
                f"F{fold}", seed,
            )
            cnn_models.append(model)

        # CNN 集成 + TTA 预测
        cnn_probs, true_labels = predict_cnn_tta(
            cnn_models, va_ds, device, n_tta
        )

        cnn_acc = accuracy_score(true_labels, cnn_probs.argmax(axis=1))
        logger.info(f"  CNN 集成+TTA: Acc = {cnn_acc:.4f}")
        fold_cnn_acc.append(cnn_acc)

        # ============================================================
        #  1b. GradientBoosting (SMOTE 增强)
        # ============================================================
        logger.info("--- GradientBoosting + SMOTE ---")

        X_train = extract_ml_features(
            tr_fps, tr_mts, energy_windows, spectrum_length, stats
        )
        X_val = extract_ml_features(
            va_fps, va_mts, energy_windows, spectrum_length, stats
        )
        y_train = np.array(tr_lbs)
        y_val = np.array(va_lbs)

        # 标准化
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # SMOTE 在标准化后的特征空间中过采样
        X_train_sm, y_train_sm = smote_oversample(X_train_s, y_train, k=5)
        logger.info(
            f"  SMOTE: {len(y_train)} -> {len(y_train_sm)} 样本"
        )

        gb = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8,
            random_state=config["training"]["seed"],
        )
        gb.fit(X_train_sm, y_train_sm)
        gb_probs = gb.predict_proba(X_val_s)

        gb_acc = accuracy_score(y_val, gb_probs.argmax(axis=1))
        logger.info(f"  GradientBoosting: Acc = {gb_acc:.4f}")
        fold_gb_acc.append(gb_acc)

        # ============================================================
        #  1c. XGBoost (SMOTE 增强)
        # ============================================================
        logger.info("--- XGBoost + SMOTE ---")

        xgb_model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=config["training"]["seed"],
        )
        xgb_model.fit(X_train_sm, y_train_sm)
        xgb_probs = xgb_model.predict_proba(X_val_s)

        xgb_acc = accuracy_score(y_val, xgb_probs.argmax(axis=1))
        logger.info(f"  XGBoost: Acc = {xgb_acc:.4f}")
        fold_xgb_acc.append(xgb_acc)

        # ============================================================
        #  存储 out-of-fold 预测
        # ============================================================
        oof_cnn[val_idx] = cnn_probs
        oof_gb[val_idx] = gb_probs
        oof_xgb[val_idx] = xgb_probs
        oof_true[val_idx] = true_labels

        # ---- 固定权重基线 (用于对比) ----
        fixed_probs = 0.5 * cnn_probs + 0.25 * gb_probs + 0.25 * xgb_probs
        fixed_preds = fixed_probs.argmax(axis=1)
        fixed_acc = accuracy_score(true_labels, fixed_preds)
        logger.info(f"  固定权重融合 (0.5/0.25/0.25): Acc = {fixed_acc:.4f}")
        fold_fixed_acc.append(fixed_acc)

        # 释放 GPU 显存
        del cnn_models
        torch.cuda.empty_cache()

    # ================================================================
    #  Phase 2: Stacking 元学习器评估
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 2: Stacking 元学习器")
    logger.info(f"{'=' * 60}")

    # 构建元特征: 默认仅拼接三个基模型的概率预测 (N, 9)
    # 可选增强: 追加每个基模型的不确定性特征(最大概率/边际/熵)，以及测量时长。
    stack_cfg = config.get("stacking", {})
    meta_mode = stack_cfg.get("meta_features", "proba_only")
    meta_X = build_meta_features(
        oof_cnn=oof_cnn,
        oof_gb=oof_gb,
        oof_xgb=oof_xgb,
        measure_times=all_mts,
        mode=str(meta_mode),
    )
    logger.info(f"  元特征模式: {meta_mode}")
    logger.info(f"  元特征维度: {meta_X.shape}")

    # 使用独立的 K-Fold 划分评估元学习器 (不同 random_state)
    # 避免与 Phase 1 共享划分导致的间接数据泄露:
    #   Phase 1 基模型的 OOF 预测中, 某些训练样本的预测来自
    #   "见过" Phase 2 验证样本的基模型, 构成间接信息通路。
    #   使用独立划分可打破这一通路, 获得更真实的评估。
    meta_skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=config["training"]["seed"] + 100,  # 独立种子
    )
    stack_preds = np.zeros(N, dtype=np.int64)
    stack_probs = np.zeros((N, C), dtype=np.float64)
    meta_fold_id = np.zeros(N, dtype=np.int64)
    fold_stack_acc = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        meta_skf.split(all_fps, stratify_key)
    ):
        fold = fold_idx + 1

        # 训练 XGBoost 元学习器
        meta_clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_child_weight=3, subsample=0.8,
            eval_metric='mlogloss',
            random_state=config["training"]["seed"],
        )
        meta_clf.fit(meta_X[train_idx], oof_true[train_idx])
        fold_proba = meta_clf.predict_proba(meta_X[val_idx])
        fold_preds = fold_proba.argmax(axis=1)
        stack_preds[val_idx] = fold_preds
        stack_probs[val_idx] = fold_proba
        meta_fold_id[val_idx] = fold

        fold_acc = accuracy_score(oof_true[val_idx], fold_preds)
        fold_f1 = f1_score(
            oof_true[val_idx], fold_preds,
            average="macro", zero_division=0,
        )
        fold_stack_acc.append(fold_acc)

        logger.info(f"  Fold {fold}: Acc={fold_acc:.4f}  F1={fold_f1:.4f}")
        logger.info(
            f"\n{classification_report(oof_true[val_idx], fold_preds, target_names=class_names, zero_division=0)}"
        )

    # ================================================================
    #  最终汇总
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("最终结果对比")
    logger.info(f"{'=' * 60}")

    results = {
        "CNN 集成+TTA": fold_cnn_acc,
        "GradientBoosting": fold_gb_acc,
        "XGBoost": fold_xgb_acc,
        "固定权重融合": fold_fixed_acc,
        "Stacking 元学习器": fold_stack_acc,
    }

    for name, accs in results.items():
        avg = np.mean(accs)
        std = np.std(accs)
        per_fold = "  ".join(
            [f"F{i+1}={a:.4f}" for i, a in enumerate(accs)]
        )
        logger.info(f"  {name}:")
        logger.info(f"    {per_fold}")
        logger.info(f"    平均: {avg:.4f} +/- {std:.4f}")

    # 全局 Stacking 指标
    global_acc = accuracy_score(oof_true, stack_preds)
    global_f1 = f1_score(
        oof_true, stack_preds, average="macro", zero_division=0
    )
    logger.info(f"\n  Stacking 全局: Acc={global_acc:.4f}  F1={global_f1:.4f}")

    # ================================================================
    #  训练结束: 输出可视化与逐样本明细 (正确/错误清单)
    # ================================================================
    output_cfg = config.get("output", {})
    artifact_dir = output_cfg.get(
        "artifact_dir",
        os.path.join(output_cfg.get("log_dir", "experiments/logs"), "artifacts"),
    )
    auto_open = bool(output_cfg.get("auto_open_artifacts", False))
    save_stacking_oof_artifacts(
        artifact_dir=artifact_dir,
        file_paths=all_fps.tolist(),
        measure_times=all_mts.tolist(),
        y_true=oof_true,
        y_pred=stack_preds,
        y_prob=stack_probs,
        class_names=class_names,
        meta_fold=meta_fold_id,
        auto_open=auto_open,
        logger=logger,
    )
    logger.info(f"{'=' * 60}")
    logger.info("训练完成。")


if __name__ == "__main__":
    main()
