"""
快速可视化：每个标签随机抽一条能谱，叠加对比
运行方式：python tools/plot_samples.py
"""
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import VALID_TIMES, parse_filename, load_spectrum
from src.utils import load_config

config = load_config("configs/config.json")
train_dir = config["data"]["train_dir"]
class_names = config.get("data", {}).get("class_names", ["粘土", "砂土", "粉土"])
num_classes = int(config.get("data", {}).get("num_classes", len(class_names)))


def total_count_normalize(counts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """按总计数归一化(用于可视化对齐形状)。

    注意: 这不是训练流水线中的标准化方式, 仅用于快速画图对比。
    """
    x = counts.astype(np.float64)
    s = float(x.sum())
    if s <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return x / (s + eps)

# 每个标签收集文件路径
label_files = {i: [] for i in range(num_classes)}
for fname in os.listdir(train_dir):
    result = parse_filename(fname)
    if result is None:
        continue
    _, label = result
    if 0 <= int(label) < num_classes:
        label_files[int(label)].append(os.path.join(train_dir, fname))

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 图1：3类各取1条60s样本叠加对比
ax = axes[0]
for label_idx in range(num_classes):
    # 优先选 60s 的
    candidates = [f for f in label_files[label_idx] if "60s" in os.path.basename(f)]
    if not candidates:
        candidates = label_files[label_idx]
    if not candidates:
        continue
    fpath = random.choice(candidates)
    spec = load_spectrum(fpath)
    spec = total_count_normalize(spec)
    name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
    ax.plot(spec, label=name, alpha=0.8)
ax.set_title("3类能谱对比（各取1条60s样本，总计数归一化后）")
ax.set_xlabel("道道号")
ax.set_ylabel("归一化计数")
ax.legend()
ax.grid(True, alpha=0.3)

# 图2：同一标签不同采集时长对比
ax = axes[1]
label_idx = 0  # 以粘土为例
for t in ["30s", "60s", "120s", "150s"]:
    candidates = [f for f in label_files[label_idx]
                  if os.path.basename(f).startswith(t)]
    if not candidates:
        continue
    fpath = random.choice(candidates)
    spec = load_spectrum(fpath)
    spec = total_count_normalize(spec)
    ax.plot(spec, label=f"{t}", alpha=0.8)
name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
ax.set_title(f"{name}：不同采集时长对比（总计数归一化后）")
ax.set_xlabel("道道号")
ax.set_ylabel("归一化计数")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = "experiments/artifacts/spectrum_comparison.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=120)
print(f"已保存到 {save_path}")
plt.show()
