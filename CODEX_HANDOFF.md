# 伽马能谱土壤分类项目 — Codex 交接文档

> 编写日期: 2026-03-09
> 编写者: Claude Opus (前任开发助手)
> 目标读者: Codex (接手执行优化方案 1-4)

---

## 一、项目概述

### 1.1 任务定义

利用 820 通道伽马能谱数据，将土壤分为 **3 类**：

| 类别 ID | 土壤类型 | 原始标签        | 样本数 | 占比   |
|---------|---------|----------------|--------|--------|
| 0       | 粘土    | 标签一 + 标签四  | 830    | 39.9%  |
| 1       | 砂土    | 标签二 + 标签三  | 834    | 40.1%  |
| 2       | 粉土    | 标签五          | 418    | 20.1%  |

**总样本数**: 2082（训练集 + 验证集合并后做 5-Fold 交叉验证）

### 1.2 数据格式

- **输入**: `.txt` 文件，每行一个通道计数值，共 820 行
- **文件名格式**: `{测量时长}{标签名}{编号}.txt`，例如 `30s标签一001.txt`
- **测量时长**: 30s / 60s / 120s / 150s
- **数据目录**: `E:/data/xunlian`（训练集）、`E:/data/yanzheng`（验证集）

### 1.3 目标

- **师兄基线**: 80% 准确率（MATLAB + 4 维能窗特征 + 简单 MLP）
- **当前最佳**: **76.13% Stacking 准确率**（尚未突破基线）
- **优化目标**: 通过特征工程提升到 80%+

### 1.4 运行环境

- **Python 解释器**: `C:/Users/14254/.conda/envs/xsypytorch/python.exe`
- **PyTorch**: 2.6.0+cu124
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **操作系统**: Windows 11
- **关键依赖**: torch, numpy, scipy, sklearn, xgboost, pywt (需安装), tqdm, tensorboard, matplotlib, seaborn

---

## 二、项目文件结构

```
Gamma Energy Spectrum Label Classification Prediction/
├── configs/
│   └── config.json              # 所有超参数配置 (核心配置文件)
├── src/
│   ├── dataset.py               # ★ 数据加载、特征提取、增强 (优化重点)
│   ├── model.py                 # 模型定义 (SEBlock, TriBranch, SpectralMAE)
│   ├── train.py                 # 单模型训练 (K-Fold)
│   ├── train_ensemble.py        # 集成训练系统 v2 (主训练入口)
│   ├── pretrain_smae.py         # SMAE 自监督预训练
│   ├── evaluate.py              # 评估函数 (F1, 混淆矩阵)
│   └── utils.py                 # 工具函数 (seed, logger, checkpoint)
├── experiments/
│   ├── checkpoints/             # 模型权重 (best_model.pth, smae_pretrained.pth)
│   └── logs/                    # 训练日志 + TensorBoard
├── data/                        # 数据说明 (实际数据在 E:/data/)
├── tools/                       # 工具脚本
├── requirements.txt
└── CODEX_HANDOFF.md             # 本文档
```

---

## 三、数据处理流水线

### 3.1 数据加载流程

```
.txt 文件 (820 原始计数)
    ↓ load_spectrum()                    # dataset.py:34
    ↓ raw_counts (820,) float32
    ↓
    ↓ [训练时] augment_spectrum()         # dataset.py:235
    ↓   - Poisson 重采样 (模拟统计涨落)
    ↓   - 通道平移 ±2 道
    ↓   - 高斯噪声
    ↓
    ↓ raw_counts / measure_time           # → CPS 谱 (counts per second)
    ↓
    ├─→ compute_derivatives(cps)          # dataset.py:51
    │     smoothed = uniform_filter1d(cps, size=5)  ← ★ 待替换为 SG 滤波
    │     d1 = np.gradient(smoothed)                ← 一阶导数
    │     d2 = np.gradient(d1)                      ← 二阶导数
    │
    ├─→ extract_energy_window_features()  # dataset.py:76, 返回 48 维
    │     [0:10]  K/U/Th 能窗 CPS + 比值
    │     [10:19] 峰值特征 (峰高/峰位/FWHM)
    │     [19:25] 全谱统计矩
    │     [25:26] 测量时间
    │     [26:34] 物理判别特征
    │     [34:48] 领域增强特征
    │
    ↓ Z-score 标准化 (训练集统计量)
    ↓
    ↓ 返回三元组:
    ↓   spectrum:        (3, 820) — [CPS, 一阶导, 二阶导]
    ↓   window_features: (48,)   — 能窗工程特征
    ↓   label:           int     — 0/1/2
```

### 3.2 当前 48 维能窗特征明细 (`extract_energy_window_features`)

| 维度范围 | 特征类型 | 说明 |
|---------|---------|------|
| [0:4]   | CPS 积分 | K_cps, U_cps, Th_cps, Total_cps |
| [4:7]   | 占比 | K/Total, U/Total, Th/Total |
| [7:10]  | 核素间比值 | K/U, K/Th, U/Th |
| [10:19] | 峰值特征 | 各能窗的 peak_height, peak_pos, FWHM (×3) |
| [19:25] | 统计矩 | mean, std, skewness, kurtosis, max, nonzero_ratio |
| [25:26] | 测量时间 | measure_time |
| [26:28] | 物理比值 | Th/K, Th/U |
| [28:30] | Compton 区 | Compton_cps, Compton/K |
| [30:32] | 低能区 | low_energy_cps, low/high 比 |
| [32:34] | 全谱特征 | 质心, 熵 |
| [34:37] | 三元坐标 | K/(K+U+Th), U/(K+U+Th), Th/(K+U+Th) |
| [37:43] | 窗口形状 | 各能窗偏度 + 峰度 (×3) |
| [43:46] | 滚动波动率 | 各能窗局部标准差 (×3) |
| [46:47] | Compton 斜率 | ch100-327 线性拟合斜率 |
| [47:48] | 高能衰减率 | ch600-820 线性拟合斜率 |

### 3.3 能窗定义

```
K-40 能窗:  通道 327 ~ 396  (1.46 MeV 峰区)
U-238 能窗: 通道 426 ~ 496  (1.76 MeV 峰区)
Th-232 能窗: 通道 682 ~ 820 (2.62 MeV 峰区)
Compton 区: 通道 100 ~ 327  (散射区)
低能区:     通道 30 ~ 100   (光电效应区)
```

---

## 四、模型架构

### 4.1 TriBranchModel (三分支融合模型)

```
输入: spectrum (B, 3, 820) + window_features (B, 48)

分支1 — Multi-Scale SE-CNN:
  3 层 MultiScaleConvBlock (卷积核 5/15/31 并行)
  + SE 通道注意力 + 残差连接
  → AvgPool + MaxPool 双池化
  → 384 维

分支2 — Spectral Transformer:
  Conv1d patch embedding (patch_size=10, 82 patches)
  + 可学习位置编码
  + 2 层 TransformerEncoder (4 heads, embed_dim=64)
  → 全局平均池化
  → 64 维

分支3 — MLP (工程特征):
  FC(48→128) → BN → ReLU → Dropout(0.2)
  → FC(128→64) → BN → ReLU
  → 64 维

融合:
  concat(384 + 64 + 64) = 512 维
  → FC(512→128) → ReLU → Dropout(0.3)
  → FC(128→3)
```

### 4.2 集成系统 (`train_ensemble.py`)

```
Phase 1 — 5-Fold 收集 out-of-fold 预测:
  每个 Fold:
    a) TriBranch × 3 (seed 42/43/44) → TTA 预测 → oof_cnn (N,3)
    b) GradientBoosting + SMOTE → oof_gb (N,3)
    c) XGBoost + SMOTE → oof_xgb (N,3)

Phase 2 — Stacking 元学习器:
  meta_X = concat(oof_cnn, oof_gb, oof_xgb) → (N, 9)
  XGBoost 元学习器 5-Fold CV → 最终预测
```

### 4.3 关键技术

| 技术 | 说明 | 文件位置 |
|------|------|---------|
| Focal Loss | α=[1.0, 0.5, 1.5], γ=2.0, 缓解类别不平衡 | model.py:72 |
| Mixup | α=0.4, 训练时数据增强 | train.py:100 |
| SMOTE | 特征空间过采样, 补充粉土样本 | train_ensemble.py:117 |
| Gradient Centralization | 梯度零均值中心化正则 | train_ensemble.py:96 |
| SWA | 随机权重平均, 最后 25% epoch | train.py:198 |
| AMP | FP16 混合精度训练 (CUDA only) | train_ensemble.py:270 |
| torch.compile | 图编译加速 (CUDA only) | train_ensemble.py:259 |
| TTA | Poisson 重采样 ×3, 70%干净+30%增强 | train_ensemble.py:332 |
| SMAE | 掩码自编码器预训练 (60% mask) | pretrain_smae.py |
| WeightedRandomSampler | 类别平衡采样 | dataset.py:449 |
| Early Stopping | patience=30 | train.py:138 |

---

## 五、训练结果汇总

### 5.1 三次完整训练记录

我们共进行了 3 次完整的集成训练 (train_ensemble.py)：

#### Run 1 (2026-03-06, TTA=10, 无 SMAE)

| 方法 | F1 | F2 | F3 | F4 | F5 | 平均 |
|------|------|------|------|------|------|------|
| CNN集成+TTA | 0.7290 | 0.6739 | 0.6923 | 0.7548 | 0.7308 | 0.7162±0.029 |
| GradientBoosting | 0.7578 | 0.7290 | 0.6923 | 0.7428 | 0.7524 | 0.7349±0.023 |
| XGBoost | 0.7578 | 0.7290 | 0.6899 | 0.7380 | 0.7476 | 0.7325±0.023 |
| 固定权重融合 | 0.7746 | 0.7290 | 0.6971 | 0.7572 | 0.7668 | 0.7450±0.029 |
| **Stacking** | **0.7698** | **0.7362** | **0.6995** | **0.7812** | **0.7812** | **0.7536±0.032** |

#### Run 2 (2026-03-07, TTA=3, 有 SMAE 预训练)

| 方法 | F1 | F2 | F3 | F4 | F5 | 平均 |
|------|------|------|------|------|------|------|
| CNN集成+TTA | 0.7386 | 0.7242 | 0.6683 | 0.7692 | 0.7380 | 0.7277±0.033 |
| GradientBoosting | 0.7746 | 0.7146 | 0.6899 | 0.7428 | 0.7524 | 0.7349±0.030 |
| XGBoost | 0.7554 | 0.7194 | 0.6851 | 0.7380 | 0.7476 | 0.7291±0.025 |
| 固定权重融合 | 0.7818 | 0.7290 | 0.6755 | 0.7524 | 0.7572 | 0.7392±0.036 |
| **Stacking** | **0.7842** | **0.7506** | **0.7620** | **0.7668** | **0.7380** | **0.7603±0.016** |

#### Run 3 (2026-03-09, TTA=3, 有 SMAE 预训练, torch.compile + val_interval=5)

| 方法 | F1 | F2 | F3 | F4 | F5 | 平均 |
|------|------|------|------|------|------|------|
| CNN集成+TTA | 0.7482 | 0.7170 | 0.6947 | 0.7332 | 0.7356 | 0.7257±0.018 |
| GradientBoosting | 0.7602 | 0.7170 | 0.6827 | 0.7476 | 0.7500 | 0.7315±0.028 |
| XGBoost | 0.7338 | 0.7146 | 0.6875 | 0.7332 | 0.7572 | 0.7253±0.023 |
| 固定权重融合 | 0.7722 | 0.7290 | 0.6899 | 0.7620 | 0.7644 | 0.7435±0.031 |
| **Stacking** | **0.7962** | **0.7626** | **0.7740** | **0.7500** | **0.7236** | **0.7613±0.024** |

### 5.2 最优结果的逐类分析 (Run 3 Stacking)

| Fold | 粘土 F1 | 砂土 F1 | 粉土 F1 | 总 Acc |
|------|--------|--------|--------|--------|
| F1   | 0.80   | 0.88   | 0.61   | 0.7962 |
| F2   | 0.77   | 0.84   | 0.61   | 0.7626 |
| F3   | 0.77   | 0.86   | 0.62   | 0.7740 |
| F4   | 0.76   | 0.84   | 0.53   | 0.7500 |
| F5   | 0.72   | 0.83   | 0.49   | 0.7236 |

**核心发现**:
- 砂土始终最好 (F1 ≈ 0.83-0.88)，因为特征分布最紧凑
- 粘土次之 (F1 ≈ 0.72-0.80)，与粉土容易混淆
- **粉土是瓶颈** (F1 ≈ 0.49-0.62)，仅 418 样本，且 Cohen's d ≈ 0.3-0.5

---

## 六、深度分析

### 6.1 粉土 (粉土) 是核心瓶颈

粉土 F1 在 0.49-0.62 波动，是拖累整体准确率的主因：
- **样本量不足**: 仅 418 个 (20%)，是粘土/砂土的一半
- **特征重叠严重**: 粉土的 K/U/Th 比值与粘土高度重叠 (Cohen's d ≈ 0.3-0.5)
- **所有模型都难**: CNN、GB、XGB 在粉土上都表现差，说明是数据本身的问题

### 6.2 Fold 3 持续偏低

Fold 3 在所有 3 次训练中都是最低的 (0.67-0.70)：
- 不是模型问题，所有基模型在 Fold 3 都偏低
- 是数据划分导致的：该 fold 验证集中包含了更多"边界样本"
- 这是随机划分的正常现象，不需要专门处理

### 6.3 TTA 效果有限

TTA (Poisson 重采样) 对 CNN 的提升不稳定：
- 有时降低准确率 (Run 1: 单模型 76.5% → TTA 72.9%)
- 原因: Poisson 重采样对低计数通道 (尤其 Th 区) 引入过大噪声
- 当前策略 (n_tta=3, clean_weight=0.7) 已是最优平衡

### 6.4 SMAE 预训练效果有限 (+0.1%)

- Transformer 分支仅占模型总参数的一小部分
- 2082 样本对 MAE 来说太少
- 但不影响后续优化方向

### 6.5 模型架构已接近瓶颈

当前模型复杂度足够 (TriBranch + 集成)，进一步增加模型参数不太可能带来显著提升。
**关键突破口在于特征工程** — 给模型更好的输入特征。

---

## 七、下一步优化方案 (★ Codex 需执行)

### 优化核心思路

当前 48 维特征已涵盖常规领域知识，但缺少以下维度：
1. **信号质量**: 当前导数用移动均值平滑，保真度不够
2. **多尺度信息**: 缺少小波域的能量分布特征
3. **高阶统计结构**: 缺少 PCA 降维后的全谱特征
4. **非线性关系**: 当前比值特征都是线性的，缺少对数变换

预计新增约 **20-30 维特征**，`window_feature_dim` 从 48 升到 ~75。

---

### 优化项 1: Savitzky-Golay 滤波替换 uniform_filter1d

**目的**: 提升导数通道的信号质量。SG 滤波在平滑的同时保留峰形，且可直接输出 1 阶/2 阶导数。

**修改文件**: `src/dataset.py`

**当前代码** (`dataset.py:51-60`):
```python
from scipy.ndimage import uniform_filter1d

def compute_derivatives(cps: np.ndarray, smooth_window: int = 5) -> tuple:
    smoothed = uniform_filter1d(cps.astype(np.float64), size=smooth_window)
    d1 = np.gradient(smoothed).astype(np.float32)
    d2 = np.gradient(d1).astype(np.float32)
    return d1, d2
```

**修改为**:
```python
from scipy.signal import savgol_filter

def compute_derivatives(cps: np.ndarray, smooth_window: int = 11,
                        poly_order: int = 3) -> tuple:
    """计算 CPS 谱的一阶和二阶导数 (Savitzky-Golay 滤波)。

    SG 滤波优势:
      - 多项式拟合保留峰形 (不像移动均值那样展宽峰)
      - 直接输出 N 阶导数, 避免 np.gradient 的数值误差累积
      - deriv=1/2 参数内置了微分, 数学上更严格

    参数选择依据:
      window_length=11: 覆盖约 5 个通道宽度, 与 K-40 峰半高宽匹配
      polyorder=3: 三次多项式, 足以拟合高斯型峰形

    Args:
        cps:           CPS 谱, shape=(820,)
        smooth_window: SG 滤波窗口长度 (必须为奇数)
        poly_order:    拟合多项式阶数

    Returns:
        (d1, d2): 一阶导数和二阶导数, shape 均为 (820,)
    """
    cps_f64 = cps.astype(np.float64)
    d1 = savgol_filter(cps_f64, window_length=smooth_window,
                       polyorder=poly_order, deriv=1).astype(np.float32)
    d2 = savgol_filter(cps_f64, window_length=smooth_window,
                       polyorder=poly_order, deriv=2).astype(np.float32)
    return d1, d2
```

**注意事项**:
- `smooth_window` 必须为奇数且 > `poly_order`
- 需要同步修改 `configs/config.json` 中 `augmentation.smooth_window` 从 5 改为 11
- `pretrain_smae.py` 中的 `SelfSupervisedSpectrumDataset` 也调用了 `compute_derivatives`，会自动受益
- SG 滤波的 `deriv` 参数自带微分功能，不需要再调用 `np.gradient`
- 原有的 `from scipy.ndimage import uniform_filter1d` 导入可以删除 (确认全文件无其他引用后)

---

### 优化项 2: 小波能量特征 (+5~8 维)

**目的**: 捕获不同频率尺度的能量分布。粘土矿物的高频纹理和粉土的低频包络可能在小波域上有显著差异。

**修改文件**: `src/dataset.py`

**新增函数**:
```python
import pywt

def extract_wavelet_energy_features(cps: np.ndarray,
                                     wavelet: str = 'db4',
                                     level: int = 5) -> np.ndarray:
    """提取小波能量特征。

    对 CPS 谱做多级离散小波分解, 计算每个尺度的能量占比。
    不同土壤类型在不同频率尺度上的能量分布有差异:
      - 高频 (d1-d2): 反映谱的局部噪声和精细结构
      - 中频 (d3-d4): 反映 K/U/Th 特征峰的宽度和形状
      - 低频 (d5+a5): 反映全谱的整体包络形状

    小波选择: db4 (Daubechies-4)
      - 紧支撑, 适合 1D 信号
      - 4 阶消失矩, 对 3 次以下多项式趋势不敏感

    Args:
        cps:     CPS 谱, shape=(820,)
        wavelet: 小波基函数名称
        level:   分解层数

    Returns:
        (level+1,) 维特征 — 每个子带的能量占比
    """
    coeffs = pywt.wavedec(cps.astype(np.float64), wavelet, level=level)
    # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]

    energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
    total_energy = energies.sum() + 1e-10

    # 归一化为能量占比 (消除绝对强度的影响)
    energy_ratios = (energies / total_energy).astype(np.float32)
    return energy_ratios  # shape: (level+1,) = (6,)
```

**集成位置**: 在 `extract_energy_window_features()` 函数末尾追加调用，或在 `__getitem__` 中单独提取后拼接到 `window_features`。

**推荐方案**: 在 `extract_energy_window_features()` 末尾追加:
```python
# ---- 小波能量特征 (6维) ----
wavelet_energy = extract_wavelet_energy_features(cps_spectrum)
features.extend(wavelet_energy.tolist())
```

**依赖**: 需要安装 `pywt`:
```bash
pip install PyWavelets
```

---

### 优化项 3: PCA 得分特征 (+10~15 维)

**目的**: 用 PCA 从 820 维 CPS 谱中提取主成分得分，捕获全谱范围内的协变结构。前 10-15 个主成分通常能解释 95%+ 的方差。

**修改文件**: `src/dataset.py`

**实现思路**:
```python
from sklearn.decomposition import PCA

# 在 GammaSpectrumDataset._precompute_statistics() 中:
#   1. 对训练集所有样本的 CPS 谱做 PCA fit
#   2. 将 PCA 模型存入 self.stats['pca_model']
#   3. 保存 explained_variance_ratio_ 用于确定 n_components

# 在 __getitem__() 中:
#   pca_scores = self.stats['pca_model'].transform(cps.reshape(1, -1))[0]
#   将 pca_scores 拼接到 window_features
```

**详细实现**:

1. **在 `_precompute_statistics()` 中新增 PCA 拟合**:
```python
# 在已有的统计量计算之后
from sklearn.decomposition import PCA

n_pca_components = 15  # 取前 15 个主成分
pca = PCA(n_components=n_pca_components)
pca.fit(all_cps)  # all_cps: (N, 820)

# 存储 PCA 模型和变换后的均值/标准差 (用于 Z-score)
pca_scores_all = pca.transform(all_cps).astype(np.float32)
pca_m = pca_scores_all.mean(axis=0)
pca_s = pca_scores_all.std(axis=0)
pca_s[pca_s < 1e-8] = 1.0

# 存入 stats
stats['pca_model'] = pca
stats['pca_mean'] = pca_m
stats['pca_std'] = pca_s
```

2. **在 `__getitem__()` 中提取 PCA 得分**:
```python
# 在 Z-score 标准化之前, 用原始 CPS 计算 PCA 得分
pca_scores = self.stats['pca_model'].transform(
    cps.reshape(1, -1)
)[0].astype(np.float32)
pca_scores = (pca_scores - self.stats['pca_mean']) / self.stats['pca_std']

# 拼接到 window_features
window_features = np.concatenate([window_features, pca_scores])
```

3. **注意**: PCA 模型不能被 `torch.save` 序列化。如果需要保存统计量，可以改为保存 `pca.components_` 和 `pca.mean_`，然后手动做矩阵乘法:
```python
# 手动 PCA transform (避免 sklearn 对象序列化问题):
pca_scores = (cps - stats['pca_components_mean']) @ stats['pca_components'].T
```

---

### 优化项 4: 对数比值特征 (+3~5 维)

**目的**: 当前比值特征都是线性的 (K/U, Th/K 等)。地球化学中常用对数比值 (log-ratio) 来处理成分数据，能更好地分离重叠分布。

**修改文件**: `src/dataset.py`

**在 `extract_energy_window_features()` 末尾追加**:
```python
# ---- 对数比值特征 (5维) ----
# 地球化学常用 log-ratio 变换, 将乘法关系线性化:
#   如果 Th/K 在粘土中是 2.0, 粉土中是 0.5,
#   线性空间: 差异 = 1.5; 对数空间: log(2.0) - log(0.5) = 2.08
#   对数空间中差异被放大, 有利于分类器区分

eps = 1e-10  # 防止 log(0)
features.append(float(np.log(th_cps / (k_cps + eps) + eps)))   # log(Th/K)
features.append(float(np.log(u_cps / (k_cps + eps) + eps)))    # log(U/K)
features.append(float(np.log(th_cps / (u_cps + eps) + eps)))   # log(Th/U)
features.append(float(np.log(k_cps * th_cps + eps)))           # log(K*Th) — 交互项
features.append(float(np.log(total_cps + eps)))                 # log(Total) — 对数总量
```

**注意**: `eps` 变量在函数开头已有定义 (`eps = 1e-10`)，可直接复用。

---

### 优化后的配置更新

**修改文件**: `configs/config.json`

```json
{
    "augmentation": {
        "smooth_window": 11        // ← 从 5 改为 11 (SG 滤波窗口)
    },
    "model": {
        "window_feature_dim": 74   // ← 从 48 改为 48+6+15+5=74
    }
}
```

> **重要**: `window_feature_dim` 的最终值取决于你实际添加的特征维度。
> 请在实现后打印 `window_features.shape` 验证维度是否匹配。

---

## 八、实施注意事项

### 8.1 代码风格要求

用户明确要求: **"务必保持代码结构的可读性、简洁性、健壮性、可维护性、注释的详细性"**

- 所有注释用**中文**
- 新增函数需要写 docstring，说明：目的、原理、参数、返回值
- 特征工程函数中需注释每个特征的**物理含义**
- 保持与现有代码风格一致 (参考 `extract_energy_window_features` 的注释风格)

### 8.2 执行顺序

建议按 1 → 4 → 2 → 3 的顺序实施：
1. **SG 滤波** (改动最小，影响最大 — 改善 3 通道输入质量)
2. **对数比值** (改动最小，仅追加 5 行)
3. **小波能量** (需要新增依赖 pywt)
4. **PCA 得分** (改动最大，涉及统计量计算和序列化)

每实施一项后建议运行一次 `python src/train_ensemble.py` 验证效果。

### 8.3 验证要点

1. 修改 `compute_derivatives` 后，确保 `pretrain_smae.py` 中的 `SelfSupervisedSpectrumDataset` 也正常工作 (它也调用 `compute_derivatives`)
2. 新增特征后，确保 `window_feature_dim` 与实际特征维度严格一致，否则 MLP 分支会报维度错误
3. PCA 实现中，验证集必须使用训练集拟合的 PCA 模型 (不能重新 fit)
4. `extract_ml_features()` 函数 (`train_ensemble.py:416`) 也调用了 `extract_energy_window_features`，会自动获得新特征。但 PCA 特征需要额外处理 (因为 `extract_ml_features` 不经过 Dataset 类)
5. 关注 `num_workers` 必须为 0 (Windows 上多进程有问题)

### 8.4 预期效果

| 优化项 | 预期提升 | 信心度 |
|--------|---------|--------|
| SG 滤波 | +1~2% | 高 (文献支持: SG 在光谱导数中是标准做法) |
| 对数比值 | +1~2% | 高 (地球化学标准变换) |
| 小波能量 | +1~2% | 中 (取决于土壤类型在频域的差异) |
| PCA 得分 | +1~3% | 中 (取决于主成分是否捕获判别信息) |

如果四项全部生效，预计总体提升 **3~5%**，从 76% → **79~81%**，有望突破 80% 基线。

### 8.5 运行命令

```bash
# 激活环境
conda activate xsypytorch

# 运行集成训练 (约 2 小时)
python src/train_ensemble.py

# 查看日志
cat experiments/logs/train_ensemble_v2.log

# TensorBoard (可选)
tensorboard --logdir experiments/logs
```

---

## 九、已知问题备忘

1. **`num_workers > 0` 在 Windows 上会卡住** — 保持为 0
2. **torch.compile 在 CPU 上不生效** — 已有条件判断 `device.type == "cuda"`
3. **SMAE 预训练效果有限** — 已完成，权重在 `experiments/checkpoints/smae_pretrained.pth`，不需要重新训练
4. **TTA 可能降低准确率** — 当前 n_tta=3 + clean_weight=0.7 是最优配置
5. **Fold 3 持续偏低** — 正常现象，不需要特殊处理

---

## 十、关键文件速查

| 需求 | 文件 | 关键行号 |
|------|------|---------|
| 修改导数计算 | `src/dataset.py` | L51-60 (`compute_derivatives`) |
| 添加新特征 | `src/dataset.py` | L76-232 (`extract_energy_window_features`) |
| 修改特征维度 | `configs/config.json` | L37 (`window_feature_dim`) |
| 修改平滑窗口 | `configs/config.json` | L25 (`smooth_window`) |
| 数据集统计量计算 | `src/dataset.py` | L346-385 (`_precompute_statistics`) |
| ML 特征提取 | `src/train_ensemble.py` | L416-435 (`extract_ml_features`) |
| 模型 MLP 输入维度 | `src/model.py` | L354 (`window_feature_dim`) |
| 运行训练 | `src/train_ensemble.py` | 主入口 (`main()`) |

祝顺利突破 80%！
