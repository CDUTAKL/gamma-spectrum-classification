import os
import random

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
    """从 CPS 谱提取 59 维通用工程特征（不含 PCA 得分）。

    特征分组:
      [0:10]   基础能窗特征: K/U/Th CPS, 总 CPS, 各窗口占比和比值
      [10:19]  峰值特征: 每个能窗的峰高/峰位/半高宽
      [19:25]  全谱统计矩: 均值/标准差/偏度/峰度/最大值/非零占比
      [25:26]  测量时间
      [26:34]  物理判别特征: Th/K, Th/U, Compton, 低能区, 谱质心, 谱熵
      [34:48]  领域增强特征 (新增): 三元坐标, 窗口偏度/峰度, 滚动波动率, Compton 斜率
      [48:53]  对数比值特征: log-ratio 变换后的非线性判别关系
      [53:59]  小波能量特征: 多尺度子带能量占比
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

    # 测量时间 (1维)
    if measure_time is not None:
        features.append(float(measure_time))

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
        不含 PCA 时返回 59 维，含 PCA 时返回 74 维
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

    if poisson_resample and random.random() < aug_prob:
        counts = np.random.poisson(counts.astype(np.int64)).astype(np.float32)

    cps = counts / measure_time

    if channel_shift_max > 0 and random.random() < aug_prob:
        shift = random.randint(-channel_shift_max, channel_shift_max)
        if shift != 0:
            cps = np.roll(cps, shift)
            if shift > 0:
                cps[:shift] = 0.0
            else:
                cps[shift:] = 0.0

    if gaussian_noise_std > 0 and random.random() < aug_prob:
        noise = np.random.normal(0, gaussian_noise_std * cps.mean(), size=cps.shape)
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

    def _precompute_statistics(self) -> dict:
        """预计算 CPS/导数/工程特征的统计量，并拟合训练集 PCA。"""
        print(f"  [统计量计算] 正在处理 {len(self.file_paths)} 个文件...")
        n = len(self.file_paths)
        L = self.spectrum_length
        window_dim = self.config["model"]["window_feature_dim"]
        base_feature_dim = window_dim - PCA_COMPONENTS

        all_cps = np.zeros((n, L), dtype=np.float64)
        all_d1 = np.zeros((n, L), dtype=np.float64)
        all_d2 = np.zeros((n, L), dtype=np.float64)
        all_base_wf = np.zeros((n, base_feature_dim), dtype=np.float64)

        for i in range(n):
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
                    f"基础工程特征维度不匹配：配置要求 {base_feature_dim}，"
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw_counts = load_spectrum(self.file_paths[idx], self.spectrum_length)
        measure_time = self.measure_times[idx]

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
        weights_per_class = total / (num_classes * np.maximum(counts, 1))
        sample_weights = np.array([weights_per_class[lb] for lb in self.labels])
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
