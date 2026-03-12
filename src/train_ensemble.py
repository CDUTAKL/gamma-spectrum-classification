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

import argparse
import copy
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime

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
#  CLI 参数解析
# =====================================================================

def parse_args():
    """解析命令行参数，支持 Phase 2-only 快速调参模式。"""
    parser = argparse.ArgumentParser(
        description="三分支异构集成训练系统 v2 — 支持 OOF 缓存与 Phase 2 快速调参"
    )
    parser.add_argument(
        "--oof-path", type=str, default=None,
        help="OOF 缓存文件路径 (npz 格式)。"
             "保存时为输出路径，Phase 2-only 时为输入路径。"
             "默认: experiments/artifacts/oof_cache.npz"
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="跳过 Phase 1 (基模型训练)，直接从 OOF 缓存加载并运行 Phase 2 stacking。"
             "需要已存在的 OOF 缓存文件。"
    )
    parser.add_argument(
        "--save-oof", action="store_true",
        help="Phase 1 结束后保存 OOF 预测到 npz 文件。"
             "默认在完整训练时自动开启。"
    )
    parser.add_argument(
        "--meta-features",
        type=str,
        choices=["proba_only", "proba+uncertainty", "proba+uncertainty+time", "uncertainty_only"],
        default=None,
        help="覆盖 config.json 中 stacking.meta_features 的设置。"
             "可选: proba_only, proba+uncertainty, proba+uncertainty+time"
    )
    return parser.parse_args()


# =====================================================================
#  OOF 缓存 Save / Load
# =====================================================================

def _config_hash(config: dict) -> str:
    """计算 Phase 1 配置指纹，用于验证 OOF 缓存与当前配置一致性。

    关键点：
      - 要允许 Phase 2-only 模式覆盖 stacking.meta_features
      - 要允许读取缓存中的 seed / n_splits 以复现实验
      - output 路径变化不应使缓存失效

    因此 hash 只覆盖真正会影响 Phase 1 OOF 预测内容的配置子集。
    """
    phase1_cfg = copy.deepcopy(config)
    phase1_cfg.pop("output", None)
    phase1_cfg.pop("stacking", None)

    training_cfg = dict(phase1_cfg.get("training", {}))
    training_cfg.pop("seed", None)
    training_cfg.pop("n_splits", None)
    phase1_cfg["training"] = training_cfg

    raw = json.dumps(phase1_cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def save_oof_cache(
    path: str,
    oof_cnn: np.ndarray,
    oof_gb: np.ndarray,
    oof_xgb: np.ndarray,
    oof_true: np.ndarray,
    all_mts: np.ndarray,
    all_fps: np.ndarray,
    stratify_key: np.ndarray,
    fold_cnn_acc: list,
    fold_gb_acc: list,
    fold_xgb_acc: list,
    fold_fixed_acc: list,
    config: dict,
    logger,
) -> str:
    """将 Phase 1 的 OOF 预测保存为 npz 压缩文件。

    保存内容:
      - oof_cnn, oof_gb, oof_xgb: (N, C) 基模型 OOF 概率
      - oof_true: (N,) 真实标签
      - all_mts: (N,) 测量时长
      - all_fps: (N,) 文件路径
      - stratify_key: (N,) 分层键 (label * 10 + time_bin)
      - fold_*_acc: 逐 fold 准确率列表
      - 元数据: config_hash, seed, n_splits, num_classes, created_at

    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        # === OOF 概率矩阵 ===
        oof_cnn=oof_cnn,
        oof_gb=oof_gb,
        oof_xgb=oof_xgb,
        oof_true=oof_true,
        # === 样本元信息 ===
        all_mts=all_mts,
        all_fps=all_fps,
        stratify_key=stratify_key,
        # === 逐 fold 准确率 ===
        fold_cnn_acc=np.array(fold_cnn_acc),
        fold_gb_acc=np.array(fold_gb_acc),
        fold_xgb_acc=np.array(fold_xgb_acc),
        fold_fixed_acc=np.array(fold_fixed_acc),
        # === 元数据 ===
        config_hash=np.array(_config_hash(config)),
        seed=np.array(config["training"]["seed"]),
        n_splits=np.array(config["training"].get("n_splits", 5)),
        num_classes=np.array(config["data"]["num_classes"]),
        created_at=np.array(datetime.now().isoformat()),
    )
    file_size = os.path.getsize(path) / 1024
    logger.info(f"OOF 缓存已保存: {os.path.abspath(path)} ({file_size:.1f} KB)")
    return path


def load_oof_cache(path: str, config: dict, logger) -> dict:
    """加载 OOF 缓存并验证与当前配置的一致性。

    验证项:
      - config_hash: 关键超参数是否匹配
      - num_classes, n_splits: 维度是否一致
      - oof_cnn/gb/xgb 形状: (N, C) 是否匹配

    Returns:
        包含所有 OOF 数据和元信息的字典

    Raises:
        FileNotFoundError: 缓存文件不存在
        ValueError: 缓存与当前配置不一致
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"OOF 缓存文件不存在: {path}")

    # 不需要 pickle；禁用可避免读取被污染 npz 时触发反序列化风险。
    data = np.load(path, allow_pickle=False)
    logger.info(f"加载 OOF 缓存: {os.path.abspath(path)}")

    # === 元数据验证 ===
    cached_hash = str(data["config_hash"])
    current_hash = _config_hash(config)
    if cached_hash != current_hash:
        raise ValueError(
            f"OOF 缓存配置指纹不匹配!\n"
            f"  缓存: {cached_hash}\n"
            f"  当前: {current_hash}\n"
            f"  请重新运行完整训练生成新的 OOF 缓存。"
        )

    cached_C = int(data["num_classes"])
    expected_C = config["data"]["num_classes"]
    if cached_C != expected_C:
        raise ValueError(
            f"num_classes 不匹配: 缓存={cached_C}, 配置={expected_C}"
        )

    cached_splits = int(data["n_splits"])

    # === 形状验证 ===
    oof_cnn = data["oof_cnn"]
    oof_gb = data["oof_gb"]
    oof_xgb = data["oof_xgb"]
    oof_true = data["oof_true"]
    N = len(oof_true)

    if len(data["all_fps"]) != N:
        raise ValueError(f"all_fps 长度不匹配: 期望 {N}, 实际 {len(data['all_fps'])}")
    if len(data["all_mts"]) != N:
        raise ValueError(f"all_mts 长度不匹配: 期望 {N}, 实际 {len(data['all_mts'])}")
    if len(data["stratify_key"]) != N:
        raise ValueError(
            f"stratify_key 长度不匹配: 期望 {N}, 实际 {len(data['stratify_key'])}"
        )

    for name, arr in [("oof_cnn", oof_cnn), ("oof_gb", oof_gb), ("oof_xgb", oof_xgb)]:
        if arr.shape != (N, expected_C):
            raise ValueError(
                f"{name} 形状不匹配: 期望 ({N}, {expected_C}), 实际 {arr.shape}"
            )

    created_at = str(data["created_at"])
    cached_seed = int(data["seed"])
    current_seed = int(config["training"]["seed"])
    current_splits = int(config["training"].get("n_splits", 5))

    if cached_seed != current_seed:
        logger.warning(
            f"当前配置 seed={current_seed} 与缓存 seed={cached_seed} 不一致；"
            f"Phase 2-only 将使用缓存 seed 以保持可复现。"
        )
    if cached_splits != current_splits:
        logger.warning(
            f"当前配置 n_splits={current_splits} 与缓存 n_splits={cached_splits} 不一致；"
            f"Phase 2-only 将使用缓存 n_splits 以保持可复现。"
        )

    logger.info(f"  创建时间: {created_at}")
    logger.info(f"  样本数: {N}, 类别数: {cached_C}, Folds: {cached_splits}")
    logger.info(f"  Seed: {cached_seed}, 配置指纹: {cached_hash}")

    return {
        "oof_cnn": oof_cnn,
        "oof_gb": oof_gb,
        "oof_xgb": oof_xgb,
        "oof_true": oof_true,
        "all_mts": data["all_mts"],
        "all_fps": data["all_fps"],
        "stratify_key": data["stratify_key"],
        "fold_cnn_acc": data["fold_cnn_acc"].tolist(),
        "fold_gb_acc": data["fold_gb_acc"].tolist(),
        "fold_xgb_acc": data["fold_xgb_acc"].tolist(),
        "fold_fixed_acc": data["fold_fixed_acc"].tolist(),
        "seed": cached_seed,
        "n_splits": cached_splits,
        "num_classes": cached_C,
        "created_at": created_at,
    }


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
#  GPU 能力检测与 AMP 配置
# =====================================================================

def get_gpu_compute_capability(device: torch.device) -> tuple[int, int]:
    """获取 GPU 计算能力 (major, minor)。
    
    返回: (major, minor)，如 RTX 5090 为 (9, 0)，GTX 1650 为 (7, 5)
    """
    if device.type != "cuda":
        return (0, 0)
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def should_use_amp(config: dict, device: torch.device) -> bool:
    """根据配置和 GPU 能力自动判断是否启用 AMP。
    
    规则:
      - 如果 config 中明确设置 use_amp，使用该值
      - 否则默认 True，但 sm_75 及以下 (如 GTX 1650) 默认 False
    """
    # 1. 如果配置中明确设置，使用配置值
    use_amp_cfg = config["training"].get("use_amp")
    if use_amp_cfg is not None:
        return use_amp_cfg
    
    # 2. 非 CUDA 设备不启用 AMP
    if device.type != "cuda":
        return False
    
    # 3. 检测 GPU 计算能力
    major, minor = get_gpu_compute_capability(device)
    sm = major * 10 + minor  # 如 7*10+5=75
    
    # 4. sm_75 及以下默认关闭 AMP (GTX 1650 = sm_75)
    if sm <= 75:
        return False
    
    return True


# =====================================================================
#  NaN 检测异常
# =====================================================================

class NaNDetectionError(Exception):
    """训练过程中检测到 NaN 或 Inf 时抛出此异常。"""
    pass


def check_finite(tensor: torch.Tensor, name: str = "tensor"):
    """检查 tensor 是否包含 NaN 或 Inf，若有则抛出异常。"""
    if not torch.isfinite(tensor).all():
        raise NaNDetectionError(
            f"检测到非有限值 (NaN/Inf) in {name}: "
            f"finite={tensor.isfinite().sum().item()}, "
            f"nan={tensor.isnan().sum().item()}, "
            f"inf={tensor.isinf().sum().item()}"
        )

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

    # AMP: 根据 GPU 能力和配置自动决定是否启用混合精度
    use_amp = should_use_amp(config, device)
    if device.type == "cuda":
        sm = torch.cuda.get_device_capability(device)
        logger.info(f"  GPU 计算能力: sm_{sm[0]}{sm[1]}, AMP: {use_amp}")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    patience = config["training"].get("early_stopping_patience", 30)
    min_epochs = config["training"].get("min_epochs_before_early_stop", 0)
    stopper = (
        EarlyStopping(patience=patience, min_epochs=min_epochs)
        if patience > 0 else None
    )

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
        
        # NaN 检测：验证准确率为 NaN 说明模型输出有问题
        if np.isnan(val_acc):
            raise NaNDetectionError(f"验证准确率为 NaN，模型输出可能已发散")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if stopper and stopper.step(val_acc, epoch=epoch):
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
            if batch_logits is None:
                raise RuntimeError("predict_cnn_tta 收到空模型列表，无法生成 logits")
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
    if aug_accumulated is None:
        raise RuntimeError("TTA 累积 logits 为空，请检查 n_tta 是否大于 0")
    aug_avg = aug_accumulated / n_tta

    # 第 3 步: 加权融合
    final_logits = clean_weight * clean_logits + (1 - clean_weight) * aug_avg
    check_finite(clean_logits, "clean_logits")
    check_finite(aug_avg, "aug_avg")
    check_finite(final_logits, "final_logits")

    # 恢复原始状态
    dataset.is_train = orig_is_train
    dataset.tta_mode = orig_tta_mode

    probs = torch.softmax(final_logits, dim=1).numpy()
    check_finite(torch.from_numpy(probs), "cnn_probs")
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
      - "uncertainty_only": only use 3*3 uncertainty features -> (N, 9)
    """
    mode = (mode or "proba_only").strip().lower()

    # 仅不确定性特征：不包含原始概率
    if mode == "uncertainty_only":
        parts = [
            _meta_uncertainty_features(oof_cnn),
            _meta_uncertainty_features(oof_gb),
            _meta_uncertainty_features(oof_xgb),
        ]
        return np.hstack(parts).astype(np.float32)

    base = np.hstack([oof_cnn, oof_gb, oof_xgb]).astype(np.float32)
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
    stats = feature_stats if feature_stats is not None else {}
    for fp, mt in zip(file_paths, measure_times):
        raw = load_spectrum(fp, spectrum_length)
        measure_time = float(mt)
        cps = raw / measure_time
        features = extract_engineered_features(
            cps, energy_windows, measure_time, stats
        )
        X.append(features)
    return np.array(X, dtype=np.float32)


def _default_oof_path(config: dict) -> str:
    output_cfg = config.get("output", {})
    base_dir = output_cfg.get(
        "artifact_dir",
        os.path.join(output_cfg.get("log_dir", "experiments/logs"), "artifacts"),
    )
    return os.path.join(base_dir, "oof_cache.npz")


def _build_phase2_artifact_dir(config: dict, meta_mode: str, phase2_only: bool) -> str:
    output_cfg = config.get("output", {})
    base_dir = output_cfg.get(
        "artifact_dir",
        os.path.join(output_cfg.get("log_dir", "experiments/logs"), "artifacts"),
    )
    safe_mode = str(meta_mode).replace("+", "_")
    tag = "phase2_only" if phase2_only else "phase2_full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"phase2_{tag}_{safe_mode}_{timestamp}")


def run_phase2_stacking(
    *,
    config: dict,
    logger,
    class_names: list[str],
    all_fps: np.ndarray,
    all_mts: np.ndarray,
    stratify_key: np.ndarray,
    oof_cnn: np.ndarray,
    oof_gb: np.ndarray,
    oof_xgb: np.ndarray,
    oof_true: np.ndarray,
    fold_cnn_acc: list,
    fold_gb_acc: list,
    fold_xgb_acc: list,
    fold_fixed_acc: list,
    n_splits: int,
    seed: int,
    meta_mode: str,
    phase2_only: bool,
) -> dict:
    """运行 Phase 2 stacking 评估，并输出 artifacts。"""
    N = len(oof_true)
    C = config["data"]["num_classes"]

    logger.info(f"\n{'=' * 60}")
    logger.info("Phase 2: Stacking 元学习器")
    logger.info(f"{'=' * 60}")

    meta_X = build_meta_features(
        oof_cnn=oof_cnn,
        oof_gb=oof_gb,
        oof_xgb=oof_xgb,
        measure_times=all_mts,
        mode=str(meta_mode),
    )
    logger.info(f"  元特征模式: {meta_mode}")
    logger.info(f"  元特征维度: {meta_X.shape}")

    meta_skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed + 100,
    )

    # 选择 stacking 元学习器类型（默认 XGB，可切换为 logreg 等）
    stacking_cfg = config.get("stacking", {})
    meta_learner = stacking_cfg.get("meta_learner", "xgb").lower()

    stack_preds = np.zeros(N, dtype=np.int64)
    stack_probs = np.zeros((N, C), dtype=np.float64)
    meta_fold_id = np.zeros(N, dtype=np.int64)
    fold_stack_acc = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        meta_skf.split(all_fps, stratify_key)
    ):
        fold = fold_idx + 1

        if meta_learner == "logreg":
            from sklearn.linear_model import LogisticRegression
            # 为兼容不同版本的 scikit-learn，这里不显式传 multi_class，交由默认策略处理
            meta_clf = LogisticRegression(
                solver="lbfgs",
                C=0.5,
                max_iter=1000,
                n_jobs=-1,
            )
        else:
            meta_clf = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                eval_metric='mlogloss',
                random_state=seed,
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
        if not accs:
            continue
        avg = np.mean(accs)
        std = np.std(accs)
        per_fold = "  ".join([f"F{i+1}={a:.4f}" for i, a in enumerate(accs)])
        logger.info(f"  {name}:")
        logger.info(f"    {per_fold}")
        logger.info(f"    平均: {avg:.4f} +/- {std:.4f}")

    global_acc = accuracy_score(oof_true, stack_preds)
    global_f1 = f1_score(
        oof_true, stack_preds, average="macro", zero_division=0
    )
    logger.info(f"\n  Stacking 全局: Acc={global_acc:.4f}  F1={global_f1:.4f}")

    output_cfg = config.get("output", {})
    artifact_dir = _build_phase2_artifact_dir(config, meta_mode, phase2_only)
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

    return {
        "global_acc": float(global_acc),
        "global_f1": float(global_f1),
        "artifact_dir": artifact_dir,
        "meta_mode": meta_mode,
        "fold_stack_acc": fold_stack_acc,
    }


# =====================================================================
#  主函数
# =====================================================================

def main():
    args = parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json",
    )
    config = load_config(config_path)
    if args.meta_features is not None:
        config.setdefault("stacking", {})["meta_features"] = args.meta_features

    class_names = config.get("data", {}).get("class_names") or DEFAULT_CLASS_NAMES
    if len(class_names) != config["data"]["num_classes"]:
        raise ValueError(
            f"class_names 长度({len(class_names)})与 num_classes({config['data']['num_classes']})不一致"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = os.path.join(config["output"]["log_dir"], "train_ensemble_v2.log")
    logger = get_logger("ensemble_v2", log_file=log_file)
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(
        f"运行模式: {'Phase 2-only' if args.phase2_only else '完整训练 + Phase 2'}"
    )

    all_fps, all_lbs, all_mts = [], [], []
    for d in [config["data"]["train_dir"], config["data"]["val_dir"]]:
        fps, lbs, mts, _ = scan_directory(d)
        all_fps.extend(fps)
        all_lbs.extend(lbs)
        all_mts.extend(mts)

    all_fps = np.asarray(all_fps)
    all_lbs = np.asarray(all_lbs, dtype=np.int64)
    all_mts = np.asarray(all_mts, dtype=np.float64)
    N = len(all_fps)
    C = config["data"]["num_classes"]

    logger.info(f"数据总量: {N}")
    for c in range(C):
        logger.info(f"  {class_names[c]}: {(all_lbs == c).sum()}")

    n_splits = config["training"].get("n_splits", 5)
    ensemble_seeds = [42, 43, 44]
    n_tta = 3
    energy_windows = config["data"]["energy_windows"]
    spectrum_length = config["data"]["spectrum_length"]

    logger.info(
        f"策略: TriBranch x {len(ensemble_seeds)} + GB + XGB + Stacking + TTA({n_tta})"
    )
    logger.info(f"{n_splits}-Fold 交叉验证")

    time_bins = np.array([0 if t <= 60 else 1 for t in all_mts], dtype=np.int64)
    stratify_key = all_lbs * 10 + time_bins
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config["training"]["seed"],
    )

    meta_mode = config.get("stacking", {}).get("meta_features", "proba_only")
    oof_path = args.oof_path or _default_oof_path(config)
    save_oof = bool(args.save_oof or not args.phase2_only)

    if args.phase2_only:
        cached = load_oof_cache(oof_path, config, logger)
        all_fps = np.asarray(cached["all_fps"])
        all_mts = np.asarray(cached["all_mts"], dtype=np.float64)
        stratify_key = np.asarray(cached["stratify_key"], dtype=np.int64)
        oof_cnn = np.asarray(cached["oof_cnn"], dtype=np.float64)
        oof_gb = np.asarray(cached["oof_gb"], dtype=np.float64)
        oof_xgb = np.asarray(cached["oof_xgb"], dtype=np.float64)
        oof_true = np.asarray(cached["oof_true"], dtype=np.int64)
        fold_cnn_acc = list(cached["fold_cnn_acc"])
        fold_gb_acc = list(cached["fold_gb_acc"])
        fold_xgb_acc = list(cached["fold_xgb_acc"])
        fold_fixed_acc = list(cached["fold_fixed_acc"])
        n_splits = int(cached["n_splits"])
        phase2_seed = int(cached["seed"])
        logger.info("跳过 Phase 1，直接使用缓存 OOF 进入 Phase 2。")
    else:
        logger.info(f"\n{'=' * 60}")
        logger.info("Phase 1: 训练基模型, 收集 out-of-fold 预测")
        logger.info(f"{'=' * 60}")

        oof_cnn = np.zeros((N, C), dtype=np.float64)
        oof_gb = np.zeros((N, C), dtype=np.float64)
        oof_xgb = np.zeros((N, C), dtype=np.float64)
        oof_true = np.zeros(N, dtype=np.int64)
        fold_cnn_acc = []
        fold_gb_acc = []
        fold_xgb_acc = []
        fold_fixed_acc = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_fps, stratify_key)):
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

            logger.info("--- TriBranch CNN 集成 ---")
            stats_cfg = copy.deepcopy(config)
            stats_cache = dict(stats_cfg.get("cache", {}))
            stats_cache["enabled"] = False
            stats_cfg["cache"] = stats_cache
            base_ds = GammaSpectrumDataset(stats_cfg, True, tr_fps, tr_lbs, tr_mts)
            stats = base_ds.stats
            del base_ds

            tr_ds = GammaSpectrumDataset(config, True, tr_fps, tr_lbs, tr_mts, stats)
            va_ds = GammaSpectrumDataset(config, False, va_fps, va_lbs, va_mts, stats)

            cnn_models = []
            cnn_probs = None
            true_labels = None
            # ---- CNN 训练 + 自动重试逻辑 ----
            # 策略: 第一次尝试正常训练 -> 如果 NaN 则禁用 compile -> 再 NaN 则禁用 AMP
            use_compile_orig = config["training"].get("use_compile", False)
            use_amp_orig = config["training"].get("use_amp")
            
            retry_strategies = [
                # (use_compile, use_amp, description)
                (True, None, "默认配置"),
                (False, None, "禁用 torch.compile 重试"),
                (False, False, "禁用 AMP 重试"),
            ]
            
            last_error = None
            for use_compile_val, use_amp_val, strategy_desc in retry_strategies:
                # 应用本次策略
                config["training"]["use_compile"] = use_compile_val
                if use_amp_val is not None:
                    config["training"]["use_amp"] = use_amp_val
                
                try:
                    cnn_models = []
                    for seed in ensemble_seeds:
                        model, _ = train_cnn_model(
                            config, tr_ds, va_ds, device, logger, f"F{fold}", seed
                        )
                        cnn_models.append(model)

                    # 推理阶段也可能出现 NaN/Inf; 纳入重试闭环。
                    cnn_probs, true_labels = predict_cnn_tta(
                        cnn_models, va_ds, device, n_tta
                    )
                    # 如果成功，跳出重试循环
                    break
                except NaNDetectionError as e:
                    last_error = e
                    logger.warning(f"  检测到 NaN ({strategy_desc}): {e}")
                    # 清理 GPU 缓存
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    cnn_models = []  # 清空重试列表
                    cnn_probs = None
                    true_labels = None
                    # 继续下一次重试
            else:
                # 所有策略都失败
                raise RuntimeError(
                    f"CNN 训练失败，已尝试所有重试策略: {last_error}"
                )
            
            # 恢复原始配置
            config["training"]["use_compile"] = use_compile_orig
            config["training"]["use_amp"] = use_amp_orig

            if cnn_probs is None or true_labels is None:
                raise RuntimeError("CNN 推理未产生有效输出 (cnn_probs/true_labels 为空)")
            cnn_acc = accuracy_score(true_labels, cnn_probs.argmax(axis=1))
            logger.info(f"  CNN 集成+TTA: Acc = {cnn_acc:.4f}")
            fold_cnn_acc.append(cnn_acc)

            logger.info("--- GradientBoosting + SMOTE ---")
            X_train = extract_ml_features(
                tr_fps, tr_mts, energy_windows, spectrum_length, stats
            )
            X_val = extract_ml_features(
                va_fps, va_mts, energy_windows, spectrum_length, stats
            )
            y_train = np.asarray(tr_lbs, dtype=np.int64)
            y_val = np.asarray(va_lbs, dtype=np.int64)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_train_sm, y_train_sm = smote_oversample(X_train_s, y_train, k=5)
            logger.info(f"  SMOTE: {len(y_train)} -> {len(y_train_sm)} 样本")

            gb = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=config["training"]["seed"],
            )
            gb.fit(X_train_sm, y_train_sm)
            gb_probs = gb.predict_proba(X_val_s)
            gb_acc = accuracy_score(y_val, gb_probs.argmax(axis=1))
            logger.info(f"  GradientBoosting: Acc = {gb_acc:.4f}")
            fold_gb_acc.append(gb_acc)

            logger.info("--- XGBoost + SMOTE ---")
            xgb_model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                random_state=config["training"]["seed"],
            )
            xgb_model.fit(X_train_sm, y_train_sm)
            xgb_probs = xgb_model.predict_proba(X_val_s)
            xgb_acc = accuracy_score(y_val, xgb_probs.argmax(axis=1))
            logger.info(f"  XGBoost: Acc = {xgb_acc:.4f}")
            fold_xgb_acc.append(xgb_acc)

            oof_cnn[val_idx] = cnn_probs
            oof_gb[val_idx] = gb_probs
            oof_xgb[val_idx] = xgb_probs
            oof_true[val_idx] = true_labels

            fixed_probs = 0.5 * cnn_probs + 0.25 * gb_probs + 0.25 * xgb_probs
            fixed_preds = fixed_probs.argmax(axis=1)
            fixed_acc = accuracy_score(true_labels, fixed_preds)
            logger.info(f"  固定权重融合 (0.5/0.25/0.25): Acc = {fixed_acc:.4f}")
            fold_fixed_acc.append(fixed_acc)

            del cnn_models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if save_oof:
            # ---- OOF 保存前 finite 检查 ----
            # 检查 oof_cnn 是否包含 NaN，若有则拒绝保存并报错
            cnn_finite = np.isfinite(oof_cnn).all()
            gb_finite = np.isfinite(oof_gb).all()
            xgb_finite = np.isfinite(oof_xgb).all()
            
            if not cnn_finite:
                logger.error(
                    f"oof_cnn 包含 NaN/Inf! "
                    f"finite={np.isfinite(oof_cnn).sum()}/{oof_cnn.size}, "
                    f"nan={np.isnan(oof_cnn).sum()}, inf={np.isinf(oof_cnn).sum()}"
                )
                raise ValueError("oof_cnn 包含非有限值，拒绝保存。请检查 CNN 训练过程。")
            if not gb_finite or not xgb_finite:
                logger.warning(
                    f"oof_gb 或 oof_xgb 包含 NaN/Inf，GB: {gb_finite}, XGB: {xgb_finite}"
                )
            
            logger.info(f"  OOF finite 检查通过: CNN={cnn_finite}, GB={gb_finite}, XGB={xgb_finite}")
            
            save_oof_cache(
                path=oof_path,
                oof_cnn=oof_cnn,
                oof_gb=oof_gb,
                oof_xgb=oof_xgb,
                oof_true=oof_true,
                all_mts=all_mts,
                all_fps=all_fps,
                stratify_key=stratify_key,
                fold_cnn_acc=fold_cnn_acc,
                fold_gb_acc=fold_gb_acc,
                fold_xgb_acc=fold_xgb_acc,
                fold_fixed_acc=fold_fixed_acc,
                config=config,
                logger=logger,
            )
        phase2_seed = config["training"]["seed"]
            

    run_phase2_stacking(
        config=config,
        logger=logger,
        class_names=class_names,
        all_fps=all_fps,
        all_mts=all_mts,
        stratify_key=stratify_key,
        oof_cnn=oof_cnn,
        oof_gb=oof_gb,
        oof_xgb=oof_xgb,
        oof_true=oof_true,
        fold_cnn_acc=fold_cnn_acc,
        fold_gb_acc=fold_gb_acc,
        fold_xgb_acc=fold_xgb_acc,
        fold_fixed_acc=fold_fixed_acc,
        n_splits=n_splits,
        seed=phase2_seed,
        meta_mode=str(meta_mode),
        phase2_only=args.phase2_only,
    )


if __name__ == "__main__":
    main()
