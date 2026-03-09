"""光谱掩码自编码器 (SMAE) 自监督预训练。

=== 目的 ===

  在无标签的情况下, 让 Transformer 编码器学习伽马能谱的内在结构:
  - K-40 峰区与 Compton 散射区的对应关系
  - U-238 / Th-232 峰之间的共生规律
  - 不同测量时长下的谱形不变性

  预训练后, 编码器权重迁移到 TriBranchModel 的 Transformer 分支,
  为下游分类任务提供更好的初始化 (预期提升 3-5%)。

=== 流程 ===

  1. 加载所有光谱文件 (训练集 + 验证集, 不需要标签)
  2. 构建 3 通道输入 (CPS + 一阶导 + 二阶导), Z-score 归一化
  3. 训练 SpectralMAE: 掩码 60% patches, 重建被掩码区域
  4. 保存编码器权重到 experiments/checkpoints/smae_pretrained.pth

=== 用法 ===

  python src/pretrain_smae.py

  预训练完成后, 运行 python src/train_ensemble.py 会自动加载权重。
"""

import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="The epoch parameter in `scheduler.step\\(\\)`",
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.dataset import (
    compute_derivatives,
    load_spectrum,
    scan_directory,
)
from src.model import SpectralMAE
from src.utils import get_logger, load_config, set_seed


# =====================================================================
#  自监督数据集: 仅需光谱, 不需要标签
# =====================================================================

class SelfSupervisedSpectrumDataset(Dataset):
    """自监督预训练数据集。

    与 GammaSpectrumDataset 的区别:
      - 不需要标签 (返回值只有光谱, 无 label)
      - 每次 __getitem__ 都做 Poisson 重采样 (数据增强)
      - 统计量来自外部传入, 保持与下游任务一致
    """

    def __init__(self, file_paths: list, measure_times: list,
                 spectrum_length: int = 820, smooth_window: int = 11,
                 stats: dict = None, augment: bool = True):
        """
        Args:
            file_paths:     光谱文件路径列表
            measure_times:  测量时长列表
            spectrum_length: 谱长度
            smooth_window:  导数平滑窗口
            stats:          Z-score 统计量 (若 None 则自动计算)
            augment:        是否做 Poisson 重采样增强
        """
        self.file_paths = list(file_paths)
        self.measure_times = list(measure_times)
        self.spectrum_length = spectrum_length
        self.smooth_window = smooth_window
        self.augment = augment

        if stats is not None:
            self.stats = stats
        else:
            self.stats = self._compute_stats()

    def _compute_stats(self) -> dict:
        """预计算 CPS / 一阶导 / 二阶导的 mean 和 std。"""
        print(f"  [SMAE 统计量] 正在处理 {len(self.file_paths)} 个文件...")
        n = len(self.file_paths)
        L = self.spectrum_length

        all_cps = np.zeros((n, L), dtype=np.float64)
        all_d1 = np.zeros((n, L), dtype=np.float64)
        all_d2 = np.zeros((n, L), dtype=np.float64)

        for i in range(n):
            raw = load_spectrum(self.file_paths[i], L)
            cps = raw / self.measure_times[i]
            d1, d2 = compute_derivatives(cps, self.smooth_window)
            all_cps[i] = cps
            all_d1[i] = d1
            all_d2[i] = d2

        def _ms(arr):
            m = arr.mean(axis=0).astype(np.float32)
            s = arr.std(axis=0).astype(np.float32)
            s[s < 1e-8] = 1.0
            return m, s

        cps_m, cps_s = _ms(all_cps)
        d1_m, d1_s = _ms(all_d1)
        d2_m, d2_s = _ms(all_d2)

        return {
            "cps_mean": cps_m, "cps_std": cps_s,
            "d1_mean": d1_m, "d1_std": d1_s,
            "d2_mean": d2_m, "d2_std": d2_s,
        }

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw = load_spectrum(self.file_paths[idx], self.spectrum_length)
        mt = self.measure_times[idx]

        # Poisson 重采样: 模拟测量的统计涨落, 每次返回不同版本
        if self.augment:
            raw = np.random.poisson(
                raw.astype(np.int64)
            ).astype(np.float32)

        cps = raw / mt
        d1, d2 = compute_derivatives(cps, self.smooth_window)

        # Z-score 标准化
        s = self.stats
        ch0 = ((cps - s["cps_mean"]) / s["cps_std"]).astype(np.float32)
        ch1 = ((d1 - s["d1_mean"]) / s["d1_std"]).astype(np.float32)
        ch2 = ((d2 - s["d2_mean"]) / s["d2_std"]).astype(np.float32)

        # (3, 820)
        spectrum = np.stack([ch0, ch1, ch2], axis=0)
        return torch.FloatTensor(spectrum)


# =====================================================================
#  预训练主函数
# =====================================================================

def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json",
    )
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = os.path.join(
        config["output"]["log_dir"], "pretrain_smae.log"
    )
    logger = get_logger("smae", log_file=log_file)
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 超参数 ----
    pretrain_epochs = 200        # 预训练轮数 (自监督可多训一些)
    lr = 1e-3                    # 较大学习率 (预训练阶段)
    batch_size = 64              # 预训练批大小 (无需梯度累积)
    mask_ratio = 0.6             # 掩码 60% 的 patches
    warmup_epochs = 20           # 学习率预热

    # ---- 加载全部数据 (无需标签) ----
    all_fps, all_mts = [], []
    for d in [config["data"]["train_dir"], config["data"]["val_dir"]]:
        fps, _, mts, _ = scan_directory(d)
        all_fps.extend(fps)
        all_mts.extend(mts)

    logger.info(f"自监督数据: {len(all_fps)} 个光谱文件")

    # ---- 构建数据集和加载器 ----
    spectrum_length = config["data"]["spectrum_length"]
    smooth_window = config.get("augmentation", {}).get("smooth_window", 11)

    dataset = SelfSupervisedSpectrumDataset(
        all_fps, all_mts, spectrum_length, smooth_window,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ---- 构建 SpectralMAE ----
    trans_cfg = config["model"].get("transformer", {})
    model = SpectralMAE(
        in_channels=config["model"].get("in_channels", 3),
        spectrum_length=spectrum_length,
        patch_size=trans_cfg.get("patch_size", 10),
        embed_dim=trans_cfg.get("embed_dim", 64),
        num_heads=trans_cfg.get("num_heads", 4),
        num_encoder_layers=trans_cfg.get("num_layers", 2),
        ff_dim=trans_cfg.get("ff_dim", 128),
        dropout=trans_cfg.get("dropout", 0.1),
        mask_ratio=mask_ratio,
    ).to(device)

    # torch.compile: 图编译优化, 预计提速 15-30%
    if config["training"].get("use_compile", False) and device.type == "cuda":
        model = torch.compile(model, backend="aot_eager")
        logger.info("已启用 torch.compile(backend='aot_eager') 图编译")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SpectralMAE 参数量: {num_params:,}")
    logger.info(f"掩码比例: {mask_ratio:.0%}")
    logger.info(f"预训练轮数: {pretrain_epochs}")

    # ---- 优化器 + 学习率调度 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
    )
    # Cosine 衰减 + Warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pretrain_epochs - warmup_epochs,
        eta_min=lr * 0.01,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # AMP 混合精度
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- 训练循环 ----
    logger.info(f"\n{'=' * 50}")
    logger.info("开始 SMAE 自监督预训练")
    logger.info(f"{'=' * 50}")

    best_loss = float("inf")

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"SMAE Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _ = model(batch)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # 每 10 轮输出一次日志
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{pretrain_epochs} | "
                f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss

    # ---- 保存预训练权重 ----
    ckpt_dir = config["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, "smae_pretrained.pth")

    # 仅保存编码器权重 (解码器不需要)
    encoder_state = model.get_encoder_state_dict()
    torch.save({
        "encoder_state_dict": encoder_state,
        "mask_ratio": mask_ratio,
        "pretrain_epochs": pretrain_epochs,
        "best_loss": best_loss,
    }, save_path)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"预训练完成! 最终 Loss: {best_loss:.6f}")
    logger.info(f"编码器权重已保存到: {save_path}")
    logger.info(f"  包含 {len(encoder_state)} 个参数张量")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
