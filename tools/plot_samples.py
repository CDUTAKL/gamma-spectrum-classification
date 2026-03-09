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
from src.dataset import LABEL_MAP, VALID_TIMES, parse_filename, load_spectrum, total_count_normalize
from src.utils import load_config

config = load_config("configs/config.json")
train_dir = config["data"]["train_dir"]

# 每个标签收集文件路径
label_files = {i: [] for i in range(5)}
for fname in os.listdir(train_dir):
    result = parse_filename(fname)
    if result is None:
        continue
    _, label = result
    label_files[label].append(os.path.join(train_dir, fname))

label_names = {v: k for k, v in LABEL_MAP.items()}

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 图1：5类各取1条60s样本叠加对比
ax = axes[0]
for label_idx in range(5):
    # 优先选 60s 的
    candidates = [f for f in label_files[label_idx] if "60s" in os.path.basename(f)]
    if not candidates:
        candidates = label_files[label_idx]
    fpath = random.choice(candidates)
    spec = load_spectrum(fpath)
    spec = total_count_normalize(spec)
    ax.plot(spec, label=label_names[label_idx], alpha=0.8)
ax.set_title("5类能谱对比（各取1条60s样本，总计数归一化后）")
ax.set_xlabel("道道号")
ax.set_ylabel("归一化计数")
ax.legend()
ax.grid(True, alpha=0.3)

# 图2：同一标签不同采集时长对比
ax = axes[1]
label_idx = 0  # 以标签一为例
for t in ["30s", "60s", "120s", "150s"]:
    candidates = [f for f in label_files[label_idx]
                  if os.path.basename(f).startswith(t)]
    if not candidates:
        continue
    fpath = random.choice(candidates)
    spec = load_spectrum(fpath)
    spec = total_count_normalize(spec)
    ax.plot(spec, label=f"{t}", alpha=0.8)
ax.set_title(f"{label_names[label_idx]}：不同采集时长对比（总计数归一化后）")
ax.set_xlabel("道道号")
ax.set_ylabel("归一化计数")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = "experiments/logs/spectrum_comparison.png"
os.makedirs("experiments/logs", exist_ok=True)
plt.savefig(save_path, dpi=120)
print(f"已保存到 {save_path}")
plt.show()
