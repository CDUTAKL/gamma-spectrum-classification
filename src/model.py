import torch
import torch.nn as nn
import torch.nn.functional as F


def _remap_legacy_classifier_state_dict(
    state_dict: dict,
    backbone_len: int,
) -> dict:
    remapped = dict(state_dict)
    head_prefix = f"classifier.{backbone_len}."
    for key in list(remapped.keys()):
        if key.startswith(head_prefix):
            remapped[key.replace(head_prefix, "classifier_head.", 1)] = remapped.pop(key)
        elif key.startswith("classifier."):
            remapped[key.replace("classifier.", "fusion_backbone.", 1)] = remapped.pop(key)

    if "composition_head.weight" not in remapped and "classifier_head.weight" in remapped:
        remapped["composition_head.weight"] = remapped["classifier_head.weight"].clone()
    if "composition_head.bias" not in remapped and "classifier_head.bias" in remapped:
        remapped["composition_head.bias"] = remapped["classifier_head.bias"].clone()
    return remapped


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class MultiScaleConvBlock(nn.Module):
    """多尺度并行卷积块 + SE 注意力 + 残差连接。

    3 个不同大小的卷积核并行处理，分别捕获：
    - 窄峰特征（小核）
    - Compton 边特征（中核）
    - 整体谱形（大核）
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: list, pool_size: int, se_reduction: int = 4):
        super().__init__()
        n_branches = len(kernel_sizes)
        branch_ch = out_channels // n_branches
        # 最后一个分支吸收余数
        branch_channels = [branch_ch] * (n_branches - 1)
        branch_channels.append(out_channels - branch_ch * (n_branches - 1))

        self.branches = nn.ModuleList()
        for ch, k in zip(branch_channels, kernel_sizes):
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, ch, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ))

        self.se = SEBlock(out_channels, se_reduction)
        self.pool = nn.MaxPool1d(pool_size)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        out = self.pool(out)
        return out


class FocalLoss(nn.Module):
    """Focal Loss：自动给难分样本更高权重。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1.0 - probs).pow(self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha.unsqueeze(0).expand_as(logits)
            focal_weight = focal_weight * alpha_weight

        loss = -(focal_weight * smooth_targets * log_probs).sum(dim=1).mean()
        return loss


class DualBranchSEModel(nn.Module):
    """多尺度 SE 注意力 CNN + MLP 双分支模型。

    分支1（Multi-Scale SE-CNN）:
      多通道谱 (B, 3, 820) → 3层 MultiScaleConvBlock → 双池化(Avg+Max) → 2*C 维
      每层使用 3 个不同大小卷积核并行处理，捕获多尺度谱特征
      SE 注意力自动聚焦 K/U/Th 能窗区域

    分支2（MLP）:
      能窗特征 (B, 34) → FC(128) → BN → ReLU → Dropout → FC(64) → BN → ReLU → 64维

    融合: CNN_dim + 64 → FC → ReLU → Dropout → FC(num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]

        pool_size = model_cfg["pool_size"]
        fc_dims = model_cfg["fc_dims"]
        dropouts = model_cfg["dropouts"]
        num_classes = data_cfg["num_classes"]
        window_feature_dim = model_cfg["window_feature_dim"]
        window_hidden_dim = model_cfg["window_hidden_dim"]
        se_reduction = model_cfg.get("se_reduction", 4)
        in_channels = model_cfg.get("in_channels", 3)

        # ---- 分支1: Multi-Scale SE-CNN ----
        ms_channels = model_cfg.get("multi_scale_channels", [48, 96, 192])
        ms_kernels = model_cfg.get("multi_scale_kernels", [5, 15, 31])

        in_ch = in_channels
        conv_blocks = []
        for out_ch in ms_channels:
            conv_blocks.append(
                MultiScaleConvBlock(in_ch, out_ch, ms_kernels, pool_size, se_reduction)
            )
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*conv_blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        cnn_out_dim = ms_channels[-1] * 2  # avg + max 拼接

        # ---- 分支2: MLP with BN ----
        self.window_mlp = nn.Sequential(
            nn.Linear(window_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, window_hidden_dim),
            nn.BatchNorm1d(window_hidden_dim),
            nn.ReLU(inplace=True),
        )
        mlp_out_dim = window_hidden_dim

        # ---- 融合分类器 ----
        fused_dim = cnn_out_dim + mlp_out_dim
        fc_layers = []
        in_dim = fused_dim
        for out_dim, drop in zip(fc_dims, dropouts):
            fc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            ])
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, num_classes))
        self.fusion_backbone = nn.Sequential(*fc_layers[:-1]) if len(fc_layers) > 1 else nn.Identity()
        self.classifier_head = fc_layers[-1]
        self.composition_head = nn.Linear(in_dim, num_classes)

    def _forward_shared_features(
        self, spectrum: torch.Tensor, window_features: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_layers(spectrum)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)

        w = self.window_mlp(window_features)

        fused = torch.cat([x, w], dim=1)
        return self.fusion_backbone(fused)

    def _build_output_dict(self, shared_features: torch.Tensor) -> dict:
        logits_cls = self.classifier_head(shared_features)
        logits_reg = self.composition_head(shared_features)
        prob_cls = F.softmax(logits_cls, dim=1)
        comp_pred = F.softmax(logits_reg, dim=1)
        return {
            "logits_cls": logits_cls,
            "prob_cls": prob_cls,
            "logits_reg": logits_reg,
            "comp_pred": comp_pred,
        }

    def forward(
        self,
        spectrum: torch.Tensor,
        window_features: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        shared_features = self._forward_shared_features(spectrum, window_features)
        outputs = self._build_output_dict(shared_features)
        if return_aux:
            return outputs
        return outputs["logits_cls"]

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = _remap_legacy_classifier_state_dict(
            state_dict, backbone_len=len(self.fusion_backbone)
        )
        return super().load_state_dict(remapped, strict=strict)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
#  Spectral Transformer 分支 — 捕获全局谱段间远程物理关联
# =====================================================================

class SpectralTransformerBranch(nn.Module):
    """光谱 Transformer 分支。

    与 CNN 的局部感受野互补，通过 Self-Attention 捕获远距离谱段间的
    物理关联（如 K-40 峰区 <-> Compton 散射区 <-> Th-232 峰区）。

    架构流程:
      (B, C, 820)                              # C=3 (CPS/一阶导/二阶导)
      -> Conv1d patch embedding                # (B, embed_dim, num_patches)
      -> transpose + positional encoding       # (B, num_patches, embed_dim)
      -> N x TransformerEncoderLayer           # 多头自注意力 + FFN
      -> LayerNorm -> global average pool      # (B, embed_dim)

    参考文献:
      - ViT (Dosovitskiy et al., 2020): patch 嵌入 + Transformer 编码
      - Transformer-CNN for Soil Properties (Agronomy, 2024):
        光谱数据上 Transformer 比 ResNet18 提升 10-24% R²
    """

    def __init__(self, in_channels: int, spectrum_length: int,
                 patch_size: int = 10, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        assert spectrum_length % patch_size == 0, (
            f"spectrum_length({spectrum_length}) 必须能被 "
            f"patch_size({patch_size}) 整除"
        )
        self.num_patches = spectrum_length // patch_size  # 820/10 = 82

        # ---- Patch Embedding ----
        # 用 1D 卷积实现: kernel=stride=patch_size, 等效于线性映射每个 patch
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # ---- 可学习位置编码 ----
        # 每个 patch 位置对应一个可学习向量, 编码能量位置信息
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Transformer 编码器 ----
        # Pre-LayerNorm (norm_first=True), 训练更稳定
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ---- 输出层归一化 ----
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) 多通道谱数据, C=3, L=820
        Returns:
            (B, embed_dim) 全局谱表征向量
        """
        # Patch embedding: (B,C,820) -> (B,embed_dim,82) -> (B,82,embed_dim)
        x = self.patch_embed(x).transpose(1, 2)

        # 加位置编码: 让模型知道每个 patch 在能谱中的位置
        x = x + self.pos_embed

        # Transformer 编码: 每个 patch 通过自注意力与所有其他 patch 交互
        x = self.encoder(x)      # (B, 82, embed_dim)
        x = self.norm(x)

        # 全局平均池化: 聚合所有 patch 的表征
        x = x.mean(dim=1)        # (B, embed_dim)
        return x


# =====================================================================
#  TriBranchModel — 三分支融合模型 (本项目核心创新模型)
# =====================================================================

class ZeroBranch(nn.Module):
    """占位分支：始终输出零向量，用于在关闭 Transformer 分支时保持融合维度一致。"""
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> return (B, out_dim) 全零向量
        return x.new_zeros(x.size(0), self.out_dim)


class TriBranchModel(nn.Module):
    """三分支融合分类模型。

    三个分支从互补视角提取伽马能谱特征:

      分支1 - Multi-Scale SE-CNN:
        多尺度卷积核(5/15/31) 并行 + SE 通道注意力 + 残差连接
        -> 捕获 K/U/Th 光电峰等局部谱峰特征
        -> 输出: ms_channels[-1] x 2 = 384 维 (AvgPool + MaxPool 双池化)

      分支2 - Spectral Transformer:
        patch 嵌入 + 多头自注意力 + FFN
        -> 捕获跨能量区域的全局物理关联
        -> 输出: embed_dim = 64 维

      分支3 - MLP (工程特征):
        能窗积分 / 比值 / 峰值 / 统计矩 / 物理判别特征
        -> 编码领域先验知识
        -> 输出: window_hidden_dim = 64 维

    融合: concat(384 + 64 + 64) = 512 -> FC(128) -> Dropout -> FC(num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config["data"]
        trans_cfg = model_cfg.get("transformer", {})

        pool_size = model_cfg["pool_size"]
        fc_dims = model_cfg["fc_dims"]
        dropouts = model_cfg["dropouts"]
        num_classes = data_cfg["num_classes"]
        in_channels = model_cfg.get("in_channels", 3)
        se_reduction = model_cfg.get("se_reduction", 4)
        # 是否启用 Transformer 分支；默认启用，方便通过 config 做消融实验
        self.use_transformer_branch = model_cfg.get("use_transformer_branch", True)

        # ================ 分支1: Multi-Scale SE-CNN ================
        ms_channels = model_cfg.get("multi_scale_channels", [48, 96, 192])
        ms_kernels = model_cfg.get("multi_scale_kernels", [5, 15, 31])

        in_ch = in_channels
        conv_blocks = []
        for out_ch in ms_channels:
            conv_blocks.append(
                MultiScaleConvBlock(in_ch, out_ch, ms_kernels,
                                   pool_size, se_reduction)
            )
            in_ch = out_ch
        self.cnn_branch = nn.Sequential(*conv_blocks)
        self.cnn_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_max_pool = nn.AdaptiveMaxPool1d(1)
        cnn_out_dim = ms_channels[-1] * 2  # AvgPool + MaxPool 拼接

        # ================ 分支2: Spectral Transformer ================
        spectrum_length = data_cfg.get("spectrum_length", 820)
        self.transformer_branch = SpectralTransformerBranch(
            in_channels=in_channels,
            spectrum_length=spectrum_length,
            patch_size=trans_cfg.get("patch_size", 10),
            embed_dim=trans_cfg.get("embed_dim", 64),
            num_heads=trans_cfg.get("num_heads", 4),
            num_layers=trans_cfg.get("num_layers", 2),
            ff_dim=trans_cfg.get("ff_dim", 128),
            dropout=trans_cfg.get("dropout", 0.1),
        )
        trans_out_dim = trans_cfg.get("embed_dim", 64)
        # 记录 Transformer 分支输出维度，便于在关闭分支时构造零向量占位
        self.trans_out_dim = trans_out_dim

        # ================ 分支3: MLP (工程特征) ================
        window_feature_dim = model_cfg["window_feature_dim"]
        window_hidden_dim = model_cfg["window_hidden_dim"]

        self.mlp_branch = nn.Sequential(
            nn.Linear(window_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, window_hidden_dim),
            nn.BatchNorm1d(window_hidden_dim),
            nn.ReLU(inplace=True),
        )
        mlp_out_dim = window_hidden_dim

        # ================ 融合分类器 ================
        fused_dim = cnn_out_dim + trans_out_dim + mlp_out_dim

        fc_layers = []
        in_dim = fused_dim
        for out_dim, drop in zip(fc_dims, dropouts):
            fc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            ])
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, num_classes))
        self.fusion_backbone = nn.Sequential(*fc_layers[:-1]) if len(fc_layers) > 1 else nn.Identity()
        self.classifier_head = fc_layers[-1]
        self.composition_head = nn.Linear(in_dim, num_classes)

    def _forward_shared_features(
        self, spectrum: torch.Tensor,
        window_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spectrum:        (B, 3, 820) 多通道光谱 [CPS, 一阶导, 二阶导]
            window_features: (B, 34)     能窗工程特征
        Returns:
            (B, hidden_dim) shared features
        """
        # 分支1: CNN - 局部多尺度峰特征
        x = self.cnn_branch(spectrum)
        x_avg = self.cnn_avg_pool(x).squeeze(-1)
        x_max = self.cnn_max_pool(x).squeeze(-1)
        x_cnn = torch.cat([x_avg, x_max], dim=1)

        # 分支2: Transformer - 全局谱段关联
        if self.use_transformer_branch:
            x_trans = self.transformer_branch(spectrum)
        else:
            x_trans = x_cnn.new_zeros(x_cnn.size(0), self.trans_out_dim)

        # 分支3: MLP - 领域先验特征
        x_mlp = self.mlp_branch(window_features)

        # 三分支融合 -> 分类
        fused = torch.cat([x_cnn, x_trans, x_mlp], dim=1)
        return self.fusion_backbone(fused)

    def _build_output_dict(self, shared_features: torch.Tensor) -> dict:
        logits_cls = self.classifier_head(shared_features)
        logits_reg = self.composition_head(shared_features)
        prob_cls = F.softmax(logits_cls, dim=1)
        comp_pred = F.softmax(logits_reg, dim=1)
        return {
            "logits_cls": logits_cls,
            "prob_cls": prob_cls,
            "logits_reg": logits_reg,
            "comp_pred": comp_pred,
        }

    def forward(
        self,
        spectrum: torch.Tensor,
        window_features: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        shared_features = self._forward_shared_features(spectrum, window_features)
        outputs = self._build_output_dict(shared_features)
        if return_aux:
            return outputs
        return outputs["logits_cls"]

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = _remap_legacy_classifier_state_dict(
            state_dict, backbone_len=len(self.fusion_backbone)
        )
        return super().load_state_dict(remapped, strict=strict)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
#  SpectralMAE — 光谱掩码自编码器 (自监督预训练)
# =====================================================================

class SpectralMAE(nn.Module):
    """光谱掩码自编码器 (Spectral Masked AutoEncoder)。

    自监督预训练策略: 随机掩码光谱 patches, 训练模型重建被掩码区域。
    预训练后将编码器权重迁移到 TriBranchModel 的 Transformer 分支,
    为下游分类任务提供更好的初始化。

    === 核心思想 (来自 MAE, He et al., 2022) ===

      "学会修复缺失, 就学会了理解全局"

      对伽马能谱而言:
      - 掩码掉 K-40 峰区, 模型必须从 Compton 散射区推断 K 的存在
      - 掩码掉 Th-232 峰区, 模型必须从 U-238 峰和全谱形状推断 Th
      → 迫使编码器学习不同能量区域间的物理关联

    === 架构 ===

      编码器 (与 SpectralTransformerBranch 权重兼容):
        Conv1d patch embedding → [MASK] 替换 → 位置编码 → Transformer

      解码器 (仅预训练时使用, 微调时丢弃):
        轻量 1 层 Transformer → Linear 投影到 patch 维度

      损失:
        仅对被掩码 patches 计算 MSE (重建误差)

    参考文献:
      - MAE: Masked Autoencoders Are Scalable Vision Learners (He, 2022)
      - SMAE for Raman Spectroscopy (Expert Systems with Applications, 2025)
        在拉曼光谱上达到 83.9%, 与全监督 ResNet 持平
    """

    def __init__(self, in_channels: int = 3, spectrum_length: int = 820,
                 patch_size: int = 10, embed_dim: int = 64,
                 num_heads: int = 4, num_encoder_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1,
                 mask_ratio: float = 0.6):
        """
        Args:
            in_channels:       输入通道数 (3: CPS + 一阶导 + 二阶导)
            spectrum_length:   谱长度 (820)
            patch_size:        每个 patch 覆盖的通道数 (10)
            embed_dim:         Transformer 隐藏维度 (64)
            num_heads:         注意力头数 (4)
            num_encoder_layers: 编码器层数 (2)
            ff_dim:            FFN 中间维度 (128)
            dropout:           Dropout 比例
            mask_ratio:        掩码比例 (0.6 = 掩码 60% 的 patches)
        """
        super().__init__()

        assert spectrum_length % patch_size == 0
        self.num_patches = spectrum_length // patch_size  # 82
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # ================ 编码器 (权重可迁移到 SpectralTransformerBranch) ================

        # Patch Embedding: (B, C, 820) -> (B, 82, embed_dim)
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # 可学习位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 可学习 [MASK] 标记: 替代被掩码的 patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ================ 解码器 (仅预训练使用, 微调时丢弃) ================

        # 轻量解码器: 1 层 Transformer + 线性投影
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # 投影到原始 patch 维度: embed_dim -> in_channels * patch_size
        self.reconstruct_proj = nn.Linear(
            embed_dim, in_channels * patch_size
        )

    def _generate_mask(self, batch_size: int, device: torch.device):
        """生成随机掩码。

        Returns:
            mask: (B, num_patches) bool, True = 被掩码
        """
        num_masked = int(self.num_patches * self.mask_ratio)

        # 每个样本独立生成掩码 (不同样本掩码位置不同, 增加多样性)
        mask = torch.zeros(batch_size, self.num_patches,
                           dtype=torch.bool, device=device)
        for i in range(batch_size):
            indices = torch.randperm(self.num_patches, device=device)[:num_masked]
            mask[i, indices] = True

        return mask

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, L) 多通道光谱, C=3, L=820

        Returns:
            loss:  MSE 重建损失 (仅对被掩码 patches)
            pred:  (B, num_patches, C * patch_size) 重建结果
            mask:  (B, num_patches) 掩码位置
        """
        B = x.shape[0]

        # Step 1: 获取重建目标
        # (B, C, L) -> (B, num_patches, C * patch_size)
        target = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, 82, 10)
        target = target.permute(0, 2, 1, 3)                     # (B, 82, C, 10)
        target = target.reshape(B, self.num_patches, -1)         # (B, 82, 30)

        # Step 2: Patch Embedding
        tokens = self.patch_embed(x).transpose(1, 2)  # (B, 82, embed_dim)

        # Step 3: 掩码 — 被掩码 patches 替换为 [MASK] token
        mask = self._generate_mask(B, x.device)
        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
        tokens = torch.where(
            mask.unsqueeze(-1).expand_as(tokens),
            mask_tokens, tokens,
        )

        # Step 4: 编码器
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)

        # Step 5: 解码器
        decoded = self.decoder(encoded)
        decoded = self.decoder_norm(decoded)

        # Step 6: 重建投影
        pred = self.reconstruct_proj(decoded)  # (B, 82, 30)

        # Step 7: MSE 损失 (仅被掩码 patches)
        loss = F.mse_loss(pred[mask], target[mask])

        return loss, pred, mask

    def get_encoder_state_dict(self):
        """提取编码器权重, 用于迁移到 SpectralTransformerBranch。

        返回的 key 名称与 SpectralTransformerBranch 完全对应:
          patch_embed.weight / patch_embed.bias
          pos_embed
          encoder.*
          norm.* (encoder_norm -> norm)
        """
        state = {}
        for k, v in self.state_dict().items():
            if k.startswith('patch_embed.') or k == 'pos_embed':
                state[k] = v
            elif k.startswith('encoder.'):
                state[k] = v
            elif k.startswith('encoder_norm.'):
                # SpectralTransformerBranch 中对应 "norm"
                state[k.replace('encoder_norm.', 'norm.')] = v
        return state
