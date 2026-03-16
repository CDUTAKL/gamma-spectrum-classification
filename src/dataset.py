import os
import random
import csv

import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import pywt
except ImportError:
    pywt = None

LABEL_MAP = {"标签一": 0, "标签二": 1, "标签三": 1, "标签四": 0, "标签五": 2}
VALID_TIMES = {"30s": 30, "60s": 60, "120s": 120, "150s": 150}
WAVELET_NAME = "db4"
WAVELET_LEVEL = 5
PCA_COMPONENTS = 15


def parse_filename(filename: str):
    """从文件名解析采集时长（秒）和类别标签，失败返回 None。"""
    if not filename.endswith(".txt"):
        return None
    time_seconds = None
    for t_str, t_val in VALID_TIMES.items():
        if filename.startswith(t_str):
            time_seconds = t_val
            break
    if time_seconds is None:
        return None
    label = None
    for label_str, label_idx in LABEL_MAP.items():
        if label_str in filename:
            label = label_idx
            break
    if label is None:
        return None
    return time_seconds, label


def load_spectrum(filepath: str, spectrum_length: int = 820) -> np.ndarray:
    """读取单个 .txt 能谱文件，返回原始计数，shape=(820,)，dtype=float32。"""
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else None
    data = np.loadtxt(filepath, dtype=np.float32, delimiter=delimiter)
    if data.ndim == 1:
        counts = data
    else:
        counts = data[:, 1]
    if len(counts) != spectrum_length:
        raise ValueError(
            f"期望 {spectrum_length} 行，实际 {len(counts)} 行：{filepath}"
        )
    return counts.astype(np.float32)


def compute_derivatives(
    cps: np.ndarray,
    smooth_window: int = 11,
    poly_order: int = 3,
) -> tuple:
    """计算 CPS 谱的一阶和二阶导数（Savitzky-Golay 滤波）。

    设计目的：
      - 在平滑噪声的同时尽量保留谱峰形状，避免移动平均拉宽峰宽
      - 直接输出一阶和二阶导数，减少重复数值微分带来的误差累积

    参数约束：
      - smooth_window 必须为奇数
      - smooth_window 必须大于 poly_order
      - smooth_window 不能超过谱长

    Args:
        cps:           CPS 谱，shape=(820,)
        smooth_window: SG 滤波窗口长度
        poly_order:    局部多项式拟合阶数

    Returns:
        (d1, d2): 一阶导数和二阶导数，shape 均为 (820,)
    """
    if smooth_window % 2 == 0:
        raise ValueError(f"smooth_window 必须为奇数，当前为 {smooth_window}")
    if smooth_window <= poly_order:
        raise ValueError(
            f"smooth_window 必须大于 poly_order，当前为 {smooth_window} <= {poly_order}"
        )
    if smooth_window > len(cps):
        raise ValueError(
            f"smooth_window 不能超过谱长，当前为 {smooth_window} > {len(cps)}"
        )

    cps_f64 = cps.astype(np.float64)
    d1 = savgol_filter(
        cps_f64, window_length=smooth_window, polyorder=poly_order, deriv=1
    ).astype(np.float32)
    d2 = savgol_filter(
        cps_f64, window_length=smooth_window, polyorder=poly_order, deriv=2
    ).astype(np.float32)
    return d1, d2


def _find_peak_features(window_data: np.ndarray) -> tuple:
    """从能窗数据中提取峰值特征：峰高、峰位置、半高宽(FWHM)。"""
    if len(window_data) == 0 or window_data.max() == 0:
        return 0.0, 0.0, 0.0
    peak_height = float(window_data.max())
    peak_pos = float(np.argmax(window_data)) / max(len(window_data) - 1, 1)
    # FWHM: 半高宽
    half_max = peak_height / 2.0
    above_half = window_data >= half_max
    fwhm = float(above_half.sum()) / max(len(window_data), 1)
    return peak_height, peak_pos, fwhm


def _extract_peak_local_contrast_features(window_data: np.ndarray) -> list:
    """提取单个能窗内与峰相关的局部对比度特征（4维）。

    设计目的：
      - 峰高本身受测量强度影响大，引入“峰-背景”对比能更稳定地反映可分性
      - 对粘土/粉土等峰形相近、噪声较强的边界样本更敏感

    输出特征（按顺序）：
      1) peak_ratio:   峰高/背景（背景用窗内中位数近似），反映信噪比趋势
      2) peak_snr:     (峰高-背景)/噪声幅度（噪声用 MAD 估计），反映峰的显著性
      3) peak_offset:  峰位置相对能窗中心的偏移量（[-0.5, 0.5]），反映峰“是否跑偏”
      4) peak_focus:   峰附近局部能量占比，反映峰宽/能量集中程度
    """
    eps = 1e-10
    n = int(len(window_data))
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]

    peak_height = float(window_data.max())
    if peak_height <= 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    peak_idx = int(np.argmax(window_data))
    peak_pos = float(peak_idx) / max(n - 1, 1)

    # 背景与噪声估计：用中位数 + MAD（更稳健，避免少数尖峰拉高 std）
    bg = float(np.median(window_data))
    mad = float(np.median(np.abs(window_data - bg)))
    noise = 1.4826 * mad
    if noise < 1e-8:
        noise = float(window_data.std())
    if noise < 1e-8:
        noise = 1.0

    prominence = max(peak_height - bg, 0.0)
    peak_ratio = (peak_height + eps) / (bg + eps)
    peak_snr = prominence / (noise + eps)

    # 峰偏移：相对窗中心（0.5）偏移，中心更稳定
    peak_offset = peak_pos - 0.5

    # 峰能量集中度：以窗口长度的 10% 为半径，统计峰附近能量占比
    radius = max(2, n // 10)
    lo = max(0, peak_idx - radius)
    hi = min(n, peak_idx + radius + 1)
    local_sum = float(window_data[lo:hi].sum())
    total_sum = float(window_data.sum())
    peak_focus = local_sum / (total_sum + eps)

    return [float(peak_ratio), float(peak_snr), float(peak_offset), float(peak_focus)]


def _estimate_window_background_level(window_data: np.ndarray) -> float:
    """Estimate a simple local background level from both window edges."""
    n = int(len(window_data))
    if n == 0:
        return 0.0
    edge = max(2, n // 8)
    left = window_data[:edge]
    right = window_data[-edge:]
    bg_samples = np.concatenate([left, right], axis=0)
    return float(np.median(bg_samples))


def _estimate_window_net_area(window_data: np.ndarray) -> float:
    """Approximate net peak area after subtracting a flat local background."""
    bg_level = _estimate_window_background_level(window_data)
    net = float(np.maximum(window_data - bg_level, 0.0).sum())
    return net


def _extract_window_robust_peak_features(window_data: np.ndarray) -> list:
    """Extract robust local peak descriptors from one energy window.

    The current hard samples in this project are mostly short-duration spectra
    where the peak is weak, the background is relatively high, or the peak
    shape is not stable enough. These three features focus on that local
    structure instead of global composition only.

    Returns, in order:
      1) net_peak_snr
      2) net_peak_area_fraction
      3) peak_sharpness
    """
    eps = 1e-10
    n = int(len(window_data))
    if n == 0:
        return [0.0, 0.0, 0.0]

    window_sum = float(window_data.sum())
    peak_idx = int(np.argmax(window_data))
    peak_height = float(window_data[peak_idx])
    if peak_height <= 0.0 or window_sum <= 0.0:
        return [0.0, 0.0, 0.0]

    core_radius = max(1, n // 16)
    shoulder_radius = max(core_radius + 1, n // 8)

    core_lo = max(0, peak_idx - core_radius)
    core_hi = min(n, peak_idx + core_radius + 1)
    core = window_data[core_lo:core_hi]
    core_mean = float(core.mean()) if len(core) > 0 else peak_height

    left_shoulder = window_data[max(0, peak_idx - shoulder_radius):core_lo]
    right_shoulder = window_data[core_hi:min(n, peak_idx + shoulder_radius + 1)]
    shoulder_parts = [part for part in (left_shoulder, right_shoulder) if len(part) > 0]
    shoulder_samples = (
        np.concatenate(shoulder_parts, axis=0) if shoulder_parts else window_data
    )

    local_bg = float(np.median(shoulder_samples))
    bg_mad = float(np.median(np.abs(shoulder_samples - local_bg)))
    local_noise = 1.4826 * bg_mad
    if local_noise < 1e-8:
        local_noise = float(shoulder_samples.std())
    if local_noise < 1e-8:
        local_noise = 1.0

    net_peak_height = max(peak_height - local_bg, 0.0)
    net_peak_snr = net_peak_height / (local_noise + eps)

    net_area = float(np.maximum(window_data - local_bg, 0.0).sum())
    net_peak_area_fraction = net_area / (window_sum + eps)

    shoulder_mean = float(shoulder_samples.mean()) if len(shoulder_samples) > 0 else local_bg
    peak_sharpness = max(core_mean - shoulder_mean, 0.0) / (core_mean + eps)

    return [
        float(net_peak_snr),
        float(net_peak_area_fraction),
        float(peak_sharpness),
    ]


def extract_wavelet_energy_features(
    cps_spectrum: np.ndarray,
    wavelet: str = WAVELET_NAME,
    level: int = WAVELET_LEVEL,
) -> np.ndarray:
    """提取小波能量特征。

    设计目的：
      - 从多尺度频域角度刻画能谱结构，补充时域/能窗积分特征
      - 用各子带能量占比描述高频纹理、中频峰形和低频包络差异

    Args:
        cps_spectrum: CPS 谱，shape=(820,)
        wavelet:      小波基名称
        level:        离散小波分解层数

    Returns:
        (level + 1,) 小波子带能量占比特征
    """
    if pywt is None:
        raise ImportError(
            "当前环境缺少 PyWavelets，请先安装 `PyWavelets` 或 `pywt` 后再训练。"
        )

    coeffs = pywt.wavedec(cps_spectrum.astype(np.float64), wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
    total_energy = energies.sum() + 1e-10
    return (energies / total_energy).astype(np.float32)


def fit_pca_statistics(
    all_cps: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> dict:
    """基于训练集 CPS 谱拟合 PCA 统计量。

    设计目的：
      - 从全谱协方差结构中提取主成分得分
      - 仅使用训练集拟合，供验证集和测试时复用，避免数据泄漏

    Args:
        all_cps:       训练集全部 CPS 谱，shape=(N, 820)
        n_components:  主成分个数

    Returns:
        包含 PCA 均值、主成分矩阵和解释方差比的统计量字典
    """
    effective_components = min(n_components, all_cps.shape[0], all_cps.shape[1])
    pca = PCA(n_components=effective_components)
    pca.fit(all_cps)

    return {
        "pca_input_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def transform_pca_scores(cps_spectrum: np.ndarray, pca_stats: dict) -> np.ndarray:
    """用训练集拟合好的 PCA 统计量提取主成分得分。

    Args:
        cps_spectrum:  单条或多条 CPS 谱，shape=(820,) 或 (N, 820)
        pca_stats:     由 fit_pca_statistics 返回的统计量字典

    Returns:
        PCA 原始得分，shape=(K,) 或 (N, K)
    """
    centered = cps_spectrum.astype(np.float64) - pca_stats["pca_input_mean"]
    scores = centered @ pca_stats["pca_components"].T
    return scores.astype(np.float32)


def extract_energy_window_features(cps_spectrum: np.ndarray, energy_windows: dict,
                                    measure_time: float = None) -> np.ndarray:
    """从 CPS 谱提取 77 维通用工程特征（不含 PCA 得分）。

    特征分组:
      [0:10]   基础能窗特征: K/U/Th CPS, 总 CPS, 各窗口占比和比值
      [10:19]  峰值特征: 每个能窗的峰高/峰位/半高宽
      [19:31]  峰局部对比度特征: 每个能窗 4 维（峰/背景、SNR、中心偏移、峰集中度）
      [31:37]  全谱统计矩: 均值/标准差/偏度/峰度/最大值/非零占比
      [37:38]  测量时间
      [38:46]  物理判别特征: Th/K, Th/U, Compton, 低能区, 谱质心, 谱熵
      [46:60]  领域增强特征 (新增): 三元坐标, 窗口偏度/峰度, 滚动波动率, Compton 斜率
      [61:66]  新增粉土增强物理特征: 净峰比值 / Compton 曲率 / 主峰净面积占比
      [66:71]  对数比值特征: log-ratio 变换后的非线性判别关系
      [71:77]  小波能量特征: 多尺度子带能量占比
    """
    k_range = energy_windows["K"]
    u_range = energy_windows["U"]
    th_range = energy_windows["Th"]

    k_data = cps_spectrum[k_range[0]:k_range[1]]
    u_data = cps_spectrum[u_range[0]:u_range[1]]
    th_data = cps_spectrum[th_range[0]:th_range[1]]

    k_cps = k_data.sum()
    u_cps = u_data.sum()
    th_cps = th_data.sum()
    total_cps = cps_spectrum[:820].sum()

    net_k_cps = _estimate_window_net_area(k_data)
    net_u_cps = _estimate_window_net_area(u_data)
    net_th_cps = _estimate_window_net_area(th_data)

    eps = 1e-10
    # 基础能窗特征 (10维)
    features = [
        k_cps, u_cps, th_cps, total_cps,
        k_cps / (total_cps + eps),
        u_cps / (total_cps + eps),
        th_cps / (total_cps + eps),
        k_cps / (u_cps + eps),
        k_cps / (th_cps + eps),
        u_cps / (th_cps + eps),
    ]

    # 峰值特征：每个能窗 3 维 (peak_height, peak_pos, fwhm) → 9维
    for window_data in [k_data, u_data, th_data]:
        ph, pp, fw = _find_peak_features(window_data)
        features.extend([ph, pp, fw])

    # 峰局部对比度特征：每个能窗 4 维 → 12维
    for window_data in [k_data, u_data, th_data]:
        features.extend(_extract_peak_local_contrast_features(window_data))

    # 全谱统计矩特征 (6维)：均值、标准差、偏度、峰度、最大值、非零道占比
    spec = cps_spectrum[:820]
    spec_mean = spec.mean()
    spec_std = spec.std()
    features.append(float(spec_mean))
    features.append(float(spec_std))
    if spec_std > eps:
        skewness = float(((spec - spec_mean) ** 3).mean() / (spec_std ** 3))
    else:
        skewness = 0.0
    features.append(skewness)
    if spec_std > eps:
        kurtosis = float(((spec - spec_mean) ** 4).mean() / (spec_std ** 4) - 3.0)
    else:
        kurtosis = 0.0
    features.append(kurtosis)
    features.append(float(spec.max()))
    features.append(float((spec > eps).sum()) / 820.0)

    # 测量时间 + 全局 Poisson 噪声因子 (2维)
    # total_counts ≈ total_cps * measure_time, 相对噪声幅度 ~ 1/sqrt(total_counts)
    if measure_time is not None:
        mt = float(measure_time)
        features.append(mt)
        total_counts = total_cps * mt
        noise_factor = float(1.0 / np.sqrt(total_counts + eps))
        features.append(noise_factor)
    else:
        # 保持维度一致
        features.append(0.0)
        features.append(0.0)

    # ---- 新增物理判别特征 (8维) ----
    # Th/K 比值：经典黏土矿物判别指标
    features.append(float(th_cps / (k_cps + eps)))
    # Th/U 比值：钍铀比用于沉积物分类
    features.append(float(th_cps / (u_cps + eps)))
    # Compton 区积分 (ch 100-327)：反映体密度和平均原子序数
    compton_cps = cps_spectrum[100:327].sum()
    features.append(float(compton_cps))
    # Compton/Peak 比值
    features.append(float(compton_cps / (k_cps + eps)))
    # 低能区积分 (ch 30-100)：光电效应强度
    low_energy_cps = cps_spectrum[30:100].sum()
    features.append(float(low_energy_cps))
    # 低能/高能比
    high_energy_cps = cps_spectrum[500:820].sum()
    features.append(float(low_energy_cps / (high_energy_cps + eps)))
    # 谱质心
    channels = np.arange(820, dtype=np.float32)
    centroid = float(np.sum(spec * channels) / (total_cps + eps))
    features.append(centroid)
    # 谱熵
    prob = spec / (total_cps + eps)
    prob = prob[prob > eps]
    spectral_entropy = float(-np.sum(prob * np.log(prob)))
    features.append(spectral_entropy)

    # ---- 领域增强特征 (14维) ----
    # 三元坐标 (3维): 经典地质判别图的三端元
    #   K/(K+U+Th), U/(K+U+Th), Th/(K+U+Th)
    #   归一化到总放射性, 消除测量时长和体密度的影响
    kut_sum = k_cps + u_cps + th_cps + eps
    features.append(float(k_cps / kut_sum))
    features.append(float(u_cps / kut_sum))
    features.append(float(th_cps / kut_sum))

    # 各能窗偏度和峰度 (6维): 刻画峰形不对称性和尖锐程度
    #   粘土/粉土的矿物组成差异会导致峰形差异
    for window_data in [k_data, u_data, th_data]:
        w_mean = window_data.mean()
        w_std = window_data.std()
        if w_std > eps:
            w_skew = float(((window_data - w_mean) ** 3).mean() / (w_std ** 3))
            w_kurt = float(((window_data - w_mean) ** 4).mean() / (w_std ** 4) - 3.0)
        else:
            w_skew, w_kurt = 0.0, 0.0
        features.append(w_skew)
        features.append(w_kurt)

    # 滚动波动率 (3维): K/U/Th 窗口内 CPS 的局部标准差
    #   反映谱的"粗糙度", 与矿物颗粒大小分布相关
    for window_data in [k_data, u_data, th_data]:
        if len(window_data) >= 5:
            local_std = np.std(
                np.lib.stride_tricks.sliding_window_view(window_data, 5),
                axis=1
            ).mean()
        else:
            local_std = window_data.std()
        features.append(float(local_std))

    # Compton 斜率 (1维): 通道 100-327 的线性拟合斜率
    #   反映散射介质的平均原子序数, 黏土矿物 > 石英砂
    compton_region = cps_spectrum[100:327]
    if len(compton_region) > 1:
        x_compton = np.arange(len(compton_region), dtype=np.float64)
        # 最小二乘线性拟合斜率: slope = cov(x,y) / var(x)
        x_mean = x_compton.mean()
        y_mean = compton_region.mean()
        slope = float(
            np.sum((x_compton - x_mean) * (compton_region - y_mean))
            / (np.sum((x_compton - x_mean) ** 2) + eps)
        )
    else:
        slope = 0.0
    features.append(slope)

    # 高能区衰减率 (1维): 通道 600-820 的线性拟合斜率
    #   反映高能伽马射线的吸收特性, 与土壤含水量/密度相关
    high_region = cps_spectrum[600:820]
    if len(high_region) > 1:
        x_high = np.arange(len(high_region), dtype=np.float64)
        x_m = x_high.mean()
        y_m = high_region.mean()
        high_slope = float(
            np.sum((x_high - x_m) * (high_region - y_m))
            / (np.sum((x_high - x_m) ** 2) + eps)
        )
    else:
        high_slope = 0.0
    features.append(high_slope)

    # ---- 新增粉土增强物理特征 (5维) ----
    # 1) 净峰面积比值：主峰区先减去局部背景，再做 K/Th、Th/U、K/U 比值。
    features.append(float(net_k_cps / (net_th_cps + eps)))
    features.append(float(net_th_cps / (net_u_cps + eps)))
    features.append(float(net_k_cps / (net_u_cps + eps)))

    # 2) Compton 区曲率：高能散射背景的二次项，补充已有 slope 的一阶趋势信息。
    if len(compton_region) > 2:
        x_compton_centered = x_compton - x_compton.mean()
        compton_curvature = float(
            np.polyfit(x_compton_centered, compton_region.astype(np.float64), deg=2)[0]
        )
    else:
        compton_curvature = 0.0
    features.append(compton_curvature)

    # 3) 主峰净面积占比：三个主峰净面积在全谱中的占比，刻画“峰 vs 背景”强弱。
    net_peak_fraction = float((net_k_cps + net_u_cps + net_th_cps) / (total_cps + eps))
    features.append(net_peak_fraction)

    # ---- 局部 / 鲁棒峰特征 (9维) ----
    # 针对 persistent 粉土错分最常见的三类问题补充局部信息：
    #   - 峰是否显著高于附近背景和噪声
    #   - 窗口能量里有多少是真正的净峰而不是整体背景抬升
    #   - 峰芯是否足够尖锐、稳定，而不是宽而平的缓慢抬升
    for window_data in [k_data, u_data, th_data]:
        features.extend(_extract_window_robust_peak_features(window_data))

    # ---- 对数比值特征 (5维) ----
    # log(Th/K): 放大 Th 与 K 之间的成分差异
    features.append(float(np.log(th_cps / (k_cps + eps) + eps)))
    # log(U/K): 刻画 U 与 K 的相对富集关系
    features.append(float(np.log(u_cps / (k_cps + eps) + eps)))
    # log(Th/U): 刻画 Th 与 U 的相对富集关系
    features.append(float(np.log(th_cps / (u_cps + eps) + eps)))
    # log(K*Th): 捕获 K 与 Th 的交互强度
    features.append(float(np.log(k_cps * th_cps + eps)))
    # log(Total): 压缩总放射性强度的动态范围
    features.append(float(np.log(total_cps + eps)))

    # ---- 小波能量特征 (6维) ----
    # 各子带能量占比：刻画高频纹理、中频峰形、低频包络的能量分布
    wavelet_energy = extract_wavelet_energy_features(cps_spectrum)
    features.extend(wavelet_energy.tolist())

    return np.array(features, dtype=np.float32)


def extract_engineered_features(
    cps_spectrum: np.ndarray,
    energy_windows: dict,
    measure_time: float = None,
    feature_stats: dict = None,
) -> np.ndarray:
    """提取最终工程特征，按需追加 PCA 得分。

    Args:
        cps_spectrum:  CPS 谱，shape=(820,)
        energy_windows: 能窗配置
        measure_time:  测量时长
        feature_stats: 训练集统计量；若包含 PCA 参数则追加 PCA 得分

    Returns:
        不含 PCA 时返回 77 维，含 PCA 时返回 92 维
    """
    base_features = extract_energy_window_features(
        cps_spectrum, energy_windows, measure_time
    )

    if feature_stats is None or "pca_components" not in feature_stats:
        return base_features

    pca_scores = transform_pca_scores(cps_spectrum, feature_stats)
    return np.concatenate([base_features, pca_scores.astype(np.float32)])


def augment_spectrum(
    raw_counts: np.ndarray,
    measure_time: float,
    poisson_resample: bool = True,
    channel_shift_max: int = 2,
    aug_prob: float = 0.5,
    gaussian_noise_std: float = 0.02,
) -> np.ndarray:
    """对原始计数执行随机增强，返回增强后的 CPS 谱。"""
    counts = raw_counts.copy()

    # 时长感知增强：短时长样本本身计数低，额外噪声和通道平移适当减弱
    mt = float(measure_time)
    local_channel_shift_max = channel_shift_max
    local_gaussian_noise_std = gaussian_noise_std
    if mt <= 60.0:
        local_channel_shift_max = 0
        local_gaussian_noise_std = gaussian_noise_std * 0.5

    if poisson_resample and random.random() < aug_prob:
        counts = np.random.poisson(counts.astype(np.int64)).astype(np.float32)

    cps = counts / measure_time

    if local_channel_shift_max > 0 and random.random() < aug_prob:
        shift = random.randint(-local_channel_shift_max, local_channel_shift_max)
        if shift != 0:
            cps = np.roll(cps, shift)
            if shift > 0:
                cps[:shift] = 0.0
            else:
                cps[shift:] = 0.0

    if local_gaussian_noise_std > 0 and random.random() < aug_prob:
        noise = np.random.normal(0, local_gaussian_noise_std * cps.mean(), size=cps.shape)
        cps = np.clip(cps + noise, 0, None).astype(np.float32)

    return cps


def augment_spectrum_tta(
    raw_counts: np.ndarray,
    measure_time: float,
) -> np.ndarray:
    """TTA 专用轻量增强：仅做 Poisson 重采样。

    与训练增强不同，TTA 增强必须保持光谱的物理一致性：
    - 不做通道偏移（channel shift 会破坏能量-通道对应关系）
    - 不做高斯噪声（伽马计数本身服从 Poisson 分布，叠加高斯不合理）
    - 不做平滑（平滑会展宽特征峰，损失判别信息）

    Poisson 重采样是唯一物理上合理的增强方式：
    每个通道的计数 N 服从 Poisson(N)，重采样模拟了测量的统计涨落。

    Args:
        raw_counts: 原始计数 (820,)
        measure_time: 测量时长 (秒)

    Returns:
        CPS 谱 (820,)
    """
    resampled = np.random.poisson(
        raw_counts.astype(np.int64)
    ).astype(np.float32)
    return resampled / measure_time


def scan_directory(data_dir: str):
    """扫描目录，返回 (file_paths, labels, measure_times) 列表。"""
    file_paths, labels, measure_times = [], [], []
    skipped = 0
    for fname in sorted(os.listdir(data_dir)):
        result = parse_filename(fname)
        if result is None:
            skipped += 1
            continue
        time_seconds, label = result
        file_paths.append(os.path.join(data_dir, fname))
        labels.append(label)
        measure_times.append(time_seconds)
    return file_paths, labels, measure_times, skipped


class GammaSpectrumDataset(Dataset):
    def __init__(self, config: dict, is_train: bool = True,
                 file_paths: list = None, labels: list = None,
                 measure_times: list = None,
                 spectrum_stats: dict = None,
                 data_dir: str = None):
        """
        spectrum_stats: 预计算的统计量 dict，至少包含以下 key:
            cps_mean, cps_std, d1_mean, d1_std, d2_mean, d2_std,
            window_mean, window_std
            若启用 PCA，还包含 pca_input_mean, pca_components
        """
        self.config = config
        self.is_train = is_train
        self.tta_mode = False  # TTA 模式：仅做 Poisson 重采样
        self.spectrum_length = config["data"]["spectrum_length"]
        self.aug_cfg = config.get("augmentation", {})
        self.energy_windows = config["data"]["energy_windows"]
        self.smooth_window = config.get("augmentation", {}).get("smooth_window", 11)

        if file_paths is not None:
            self.file_paths = list(file_paths)
            self.labels = list(labels)
            self.measure_times = list(measure_times)
        elif data_dir is not None:
            fps, lbs, mts, _ = scan_directory(data_dir)
            self.file_paths = fps
            self.labels = lbs
            self.measure_times = mts
        else:
            raise ValueError("必须提供 data_dir 或 file_paths")

        if spectrum_stats is not None:
            self.stats = spectrum_stats
        else:
            self.stats = self._precompute_statistics()

        # 样本级权重（默认全为 1.0，可通过 noisy_silt_candidates.csv 做人工降权）
        self._per_sample_multipliers = np.ones(len(self.file_paths), dtype=float)
        self._load_per_sample_multipliers()

        # ---- 缓存初始化 ----
        cache_cfg = config.get("cache", {})
        self._cache_enabled = cache_cfg.get("enabled", False)
        self._l0_in_memory = cache_cfg.get("l0_in_memory", True) and self._cache_enabled

        # L0: 原始计数内存缓存 {idx: raw_counts(820,)}
        self._raw_cache: dict = {}

        # L1: 验证集完整输出缓存 {idx: (spectrum_tensor, wf_tensor)}
        self._val_cache: dict = {}
        self._val_cache_built = False

        # 预加载原始计数到内存（消除 __getitem__ 中的文件I/O）
        if self._l0_in_memory:
            self._preload_raw_counts()

        # 验证集自动构建L1缓存（确定性输出，无需每次重算）
        if not self.is_train and self._cache_enabled:
            self._build_val_cache()

    def _precompute_statistics(self) -> dict:
        """预计算 CPS/导数/工程特征的统计量，并拟合训练集 PCA。"""
        print(f"  [统计量计算] 正在处理 {len(self.file_paths)} 个文件...")
        n = len(self.file_paths)
        L = self.spectrum_length
        config_window_dim = int(self.config.get("model", {}).get("window_feature_dim", 0))
        if config_window_dim <= 0:
            raise ValueError("config.model.window_feature_dim 未设置或非法(<=0)")

        all_cps = np.zeros((n, L), dtype=np.float64)
        all_d1 = np.zeros((n, L), dtype=np.float64)
        all_d2 = np.zeros((n, L), dtype=np.float64)

        # 自动推断基础工程特征维度，并与 config 中的 window_feature_dim 做一致性校验。
        raw0 = load_spectrum(self.file_paths[0], L)
        cps0 = raw0 / self.measure_times[0]
        d1_0, d2_0 = compute_derivatives(cps0, self.smooth_window)
        base0 = extract_energy_window_features(
            cps0, self.energy_windows, self.measure_times[0]
        )
        base_feature_dim = int(base0.shape[0])
        expected_window_dim = base_feature_dim + PCA_COMPONENTS
        if config_window_dim != expected_window_dim:
            raise ValueError(
                "window_feature_dim 配置与当前特征工程实现不一致："
                f"配置为 {config_window_dim}，按实现推断应为 {expected_window_dim} "
                f"(基础 {base_feature_dim} + PCA {PCA_COMPONENTS})。"
                "请同步修改 configs/config.json -> model.window_feature_dim。"
            )
        window_dim = config_window_dim

        all_base_wf = np.zeros((n, base_feature_dim), dtype=np.float64)

        # 先写入第 0 个样本(避免重复计算)
        all_cps[0] = cps0
        all_d1[0] = d1_0
        all_d2[0] = d2_0
        all_base_wf[0] = base0

        for i in range(1, n):
            raw = load_spectrum(self.file_paths[i], L)
            cps = raw / self.measure_times[i]
            d1, d2 = compute_derivatives(cps, self.smooth_window)
            all_cps[i] = cps
            all_d1[i] = d1
            all_d2[i] = d2
            base_features = extract_energy_window_features(
                cps, self.energy_windows, self.measure_times[i]
            )
            if base_features.shape[0] != base_feature_dim:
                raise ValueError(
                    f"基础工程特征维度不一致：期望 {base_feature_dim}，"
                    f"实际提取 {base_features.shape[0]}"
                )
            all_base_wf[i] = base_features

        pca_stats = fit_pca_statistics(all_cps, n_components=PCA_COMPONENTS)
        all_pca_scores = transform_pca_scores(all_cps, pca_stats).astype(np.float64)
        all_wf = np.concatenate([all_base_wf, all_pca_scores], axis=1)
        if all_wf.shape[1] != window_dim:
            raise ValueError(
                f"最终工程特征维度不匹配：配置要求 {window_dim}，实际提取 {all_wf.shape[1]}"
            )

        def _mean_std(arr):
            m = arr.mean(axis=0).astype(np.float32)
            s = arr.std(axis=0).astype(np.float32)
            s[s < 1e-8] = 1.0
            return m, s

        cps_m, cps_s = _mean_std(all_cps)
        d1_m, d1_s = _mean_std(all_d1)
        d2_m, d2_s = _mean_std(all_d2)
        wf_m, wf_s = _mean_std(all_wf)

        return {
            "cps_mean": cps_m, "cps_std": cps_s,
            "d1_mean": d1_m, "d1_std": d1_s,
            "d2_mean": d2_m, "d2_std": d2_s,
            "window_mean": wf_m, "window_std": wf_s,
            "pca_input_mean": pca_stats["pca_input_mean"],
            "pca_components": pca_stats["pca_components"],
            "pca_explained_variance_ratio": pca_stats["pca_explained_variance_ratio"],
        }

    def _preload_raw_counts(self):
        """预加载所有原始计数到内存，消除 __getitem__ 中的文件I/O。"""
        print(f"  [L0缓存] 预加载 {len(self.file_paths)} 个原始计数到内存...")
        for i in range(len(self.file_paths)):
            self._raw_cache[i] = load_spectrum(
                self.file_paths[i], self.spectrum_length
            )
        mem_mb = len(self._raw_cache) * self.spectrum_length * 4 / 1024 / 1024
        print(f"  [L0缓存] 完成，内存占用约 {mem_mb:.1f} MB")

    def _build_val_cache(self):
        """为验证集构建完整输出缓存（仅 is_train=False 且非TTA时有效）。

        缓存内容：标准化后的 spectrum(3,820) 和 window_features(config.model.window_feature_dim,)。
        这些在验证模式下是完全确定性的，无需每个epoch重复计算。
        """
        if self._val_cache_built or self.is_train:
            return

        print(f"  [L1缓存] 构建验证集缓存 ({len(self.file_paths)} 个样本)...")
        s = self.stats

        for idx in range(len(self.file_paths)):
            # 获取原始计数（优先从L0缓存）
            if idx in self._raw_cache:
                raw_counts = self._raw_cache[idx]
            else:
                raw_counts = load_spectrum(
                    self.file_paths[idx], self.spectrum_length
                )

            measure_time = self.measure_times[idx]
            cps = raw_counts / measure_time

            d1, d2 = compute_derivatives(cps, self.smooth_window)

            window_features = extract_engineered_features(
                cps, self.energy_windows, measure_time, self.stats
            )

            # Z-score 标准化
            ch0 = ((cps - s["cps_mean"]) / s["cps_std"]).astype(np.float32)
            ch1 = ((d1 - s["d1_mean"]) / s["d1_std"]).astype(np.float32)
            ch2 = ((d2 - s["d2_mean"]) / s["d2_std"]).astype(np.float32)
            window_features = (
                (window_features - s["window_mean"]) / s["window_std"]
            ).astype(np.float32)

            spectrum = np.stack([ch0, ch1, ch2], axis=0)

            self._val_cache[idx] = (
                torch.FloatTensor(spectrum),
                torch.FloatTensor(window_features),
            )

        self._val_cache_built = True
        print(f"  [L1缓存] 完成")

    # ------------------------------------------------------------------
    #  样本级权重：基于人工审查的 noisy_silt_candidates.csv
    # ------------------------------------------------------------------

    def _load_per_sample_multipliers(self) -> None:
        """从 artifacts/noisy_silt_candidates.csv 读取 sample_weight，构建 file_path->multiplier 映射。

        设计原则：
          - 文件不存在或解析失败时静默跳过，保持默认权重 1.0；
          - 仅当 CSV 中存在与当前 file_path 完全匹配的行时才应用 sample_weight；
          - 该机制主要用于对“高置信度错分的粉土样本”做降权，不改变其它样本。
        """
        try:
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            csv_path = os.path.join(
                proj_root, "experiments", "artifacts", "noisy_silt_candidates.csv"
            )
            if not os.path.exists(csv_path):
                return

            # 兼容 UTF-8 / GBK 两种编码（Excel 在 Windows 下一般存为 GBK）
            path_to_weight = {}
            for enc in ("utf-8", "gbk"):
                try:
                    with open(csv_path, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            fp = row.get("file_path")
                            w = row.get("sample_weight", "").strip()
                            if not fp or not w:
                                continue
                            try:
                                weight = float(w)
                            except ValueError:
                                continue
                            path_to_weight[fp] = weight
                    break
                except UnicodeDecodeError:
                    path_to_weight = {}
                    continue

            if not path_to_weight:
                return

            # 按当前数据集的 file_paths 生成样本级乘子
            for i, fp in enumerate(self.file_paths):
                w = path_to_weight.get(fp)
                if w is not None:
                    self._per_sample_multipliers[i] = float(w)
        except Exception as e:
            # 出错时保持默认权重，不中断训练
            print(f"  [SampleWeights] 读取 noisy_silt_candidates.csv 失败: {e}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # ---- 快速路径：验证集L1缓存命中 ----
        if (self._cache_enabled
                and not self.is_train
                and not self.tta_mode
                and self._val_cache_built
                and idx in self._val_cache):
            spectrum, wf = self._val_cache[idx]
            return spectrum, wf, self.labels[idx]

        # ---- 获取原始计数（L0缓存或文件读取）----
        if self._l0_in_memory and idx in self._raw_cache:
            raw_counts = self._raw_cache[idx]
        else:
            raw_counts = load_spectrum(self.file_paths[idx], self.spectrum_length)

        measure_time = self.measure_times[idx]

        # ---- CPS 计算（含增强）----
        if self.tta_mode:
            # TTA 模式：仅 Poisson 重采样，保持物理一致性
            cps = augment_spectrum_tta(raw_counts, measure_time)
        elif self.is_train:
            cps = augment_spectrum(
                raw_counts, measure_time,
                poisson_resample=self.aug_cfg.get("poisson_resample", True),
                channel_shift_max=self.aug_cfg.get("channel_shift_max", 2),
                aug_prob=self.aug_cfg.get("aug_prob", 0.5),
                gaussian_noise_std=self.aug_cfg.get("gaussian_noise_std", 0.02),
            )
        else:
            cps = raw_counts / measure_time

        d1, d2 = compute_derivatives(cps, self.smooth_window)

        # 工程特征：通用特征 + 基于训练集拟合的 PCA 得分
        window_features = extract_engineered_features(
            cps, self.energy_windows, measure_time, self.stats
        )

        # Z-score 标准化（逐通道）
        s = self.stats
        ch0 = ((cps - s["cps_mean"]) / s["cps_std"]).astype(np.float32)
        ch1 = ((d1 - s["d1_mean"]) / s["d1_std"]).astype(np.float32)
        ch2 = ((d2 - s["d2_mean"]) / s["d2_std"]).astype(np.float32)
        window_features = ((window_features - s["window_mean"]) / s["window_std"]).astype(np.float32)

        # (3, 820): 原始 CPS + 一阶导数 + 二阶导数
        spectrum = np.stack([ch0, ch1, ch2], axis=0)

        return (
            torch.FloatTensor(spectrum),
            torch.FloatTensor(window_features),
            self.labels[idx],
        )

    def get_class_weights(self) -> torch.Tensor:
        num_classes = self.config["data"]["num_classes"]
        counts = np.bincount(self.labels, minlength=num_classes).astype(float)
        total = len(self.labels)
        # 基础：按 1/freq 做类平衡采样权重
        weights_per_class = total / (num_classes * np.maximum(counts, 1))
        train_cfg = self.config.get("training", {})

        # 额外 1：允许通过 training.class_sample_multipliers 对某些类（如粉土）再加权
        multipliers = train_cfg.get("class_sample_multipliers")
        if multipliers is not None and len(multipliers) == num_classes:
            multipliers = np.asarray(multipliers, dtype=float)
            weights_per_class = weights_per_class * multipliers

        # 先按类权重初始化每个样本的采样权重
        sample_weights = np.array([weights_per_class[lb] for lb in self.labels], dtype=float)

        # 额外 2：可选的按测量时长加权，对短时长样本略微提高采样权重
        # 期望配置示例：{"30": 1.2, "60": 1.05, "120": 1.0, "150": 1.0}
        time_mult_cfg = train_cfg.get("time_sample_multipliers")
        if time_mult_cfg is not None and hasattr(self, "measure_times"):
            for i, mt in enumerate(self.measure_times):
                try:
                    key = str(int(mt))
                    w_t = float(time_mult_cfg.get(key, 1.0))
                except Exception:
                    w_t = 1.0
                sample_weights[i] *= w_t

        # 额外 3：样本级降权（人工审查的 noisy_silt_candidates.csv）
        if hasattr(self, "_per_sample_multipliers"):
            sample_weights *= self._per_sample_multipliers

        return torch.FloatTensor(sample_weights)


def build_dataloaders(config: dict):
    train_dataset = GammaSpectrumDataset(
        config, is_train=True, data_dir=config["data"]["train_dir"]
    )
    val_dataset = GammaSpectrumDataset(
        config, is_train=False, data_dir=config["data"]["val_dir"],
        spectrum_stats=train_dataset.stats,
    )
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=min(num_workers, 2), pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


def build_kfold_dataloaders(config: dict, n_splits: int = 5):
    from sklearn.model_selection import StratifiedKFold

    all_fps, all_lbs, all_mts = [], [], []
    for data_dir in [config["data"]["train_dir"], config["data"]["val_dir"]]:
        fps, lbs, mts, _ = scan_directory(data_dir)
        all_fps.extend(fps)
        all_lbs.extend(lbs)
        all_mts.extend(mts)

    all_fps = np.array(all_fps)
    all_lbs = np.array(all_lbs)
    all_mts = np.array(all_mts)

    num_classes = config["data"]["num_classes"]
    class_names = config["data"].get("class_names", [f"类{i}" for i in range(num_classes)])
    print(f"\nK-Fold 数据总量: {len(all_fps)}")
    for c in range(num_classes):
        print(f"  {class_names[c]}: {(all_lbs == c).sum()}")

    # 按 (label, time_group) 联合分层，确保每个 fold 测量时长分布均匀
    time_bins = np.array([0 if t <= 60 else 1 for t in all_mts])
    stratify_key = all_lbs * 10 + time_bins

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config["training"]["seed"])
    num_workers = config["training"].get("num_workers", 0)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_fps, stratify_key)):
        train_dataset = GammaSpectrumDataset(
            config, is_train=True,
            file_paths=all_fps[train_idx].tolist(),
            labels=all_lbs[train_idx].tolist(),
            measure_times=all_mts[train_idx].tolist(),
        )
        val_dataset = GammaSpectrumDataset(
            config, is_train=False,
            file_paths=all_fps[val_idx].tolist(),
            labels=all_lbs[val_idx].tolist(),
            measure_times=all_mts[val_idx].tolist(),
            spectrum_stats=train_dataset.stats,
        )
        sample_weights = train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["batch_size"],
            sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=min(num_workers, 2), pin_memory=True, drop_last=False,
        )
        print(f"  Fold {fold_idx + 1}: 训练 {len(train_dataset)} / 验证 {len(val_dataset)}")
        folds.append((train_loader, val_loader))

    return folds
