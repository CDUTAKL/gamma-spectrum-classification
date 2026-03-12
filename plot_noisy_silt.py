import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 配置 ----------
# 最新一次完整训练的目录名（你刚才用的是这个）
ARTIFACT_DIR_NAME = "phase2_phase2_full_proba_uncertainty_20260312_001357"

# 本机数据根目录
DATA_ROOT = Path(r"E:/data")

# 每类用多少条“预测正确”的样本来做平均模板
N_TEMPLATES_PER_CLASS = 20

# 最多画多少个候选样本（避免一次弹出 50 个窗口太乱，可以先看前 10 个）
MAX_CANDIDATES = 10
  # -------------------------

proj_root = Path(__file__).resolve().parent
artifacts_dir = proj_root / "experiments" / "artifacts"
pred_dir = artifacts_dir / ARTIFACT_DIR_NAME

pred = pd.read_csv(pred_dir / "stacking_oof_predictions.csv", encoding="utf-8")
# noisy_silt_candidates.csv 是你在本机用 Excel 编辑保存的，通常是 GBK/ANSI 编码，
# 这里显式用 gbk 读取，避免 utf-8 解码错误。
cands = pd.read_csv(artifacts_dir / "noisy_silt_candidates.csv", encoding="gbk")

def to_local(path_str: str) -> Path:
    """
    将预测文件里的 file_path 映射到本机路径，兼容两种情况：
    1) 远程路径：/.../gamma_data/...  ->  E:/data/...
    2) 已经是本机路径（例如 E:/data/... 或 E:\\data\\...）
    """
    path_str = str(path_str)
    if "gamma_data/" in path_str:
        rel = path_str.split("gamma_data/")[1]
        return DATA_ROOT / rel
    return Path(path_str)


def load_spec(path_str: str) -> np.ndarray:
    """
    读取单条能谱，尽量与 src/dataset.py 中 load_spectrum 的逻辑保持一致：
    - 首行含逗号则按 CSV 读取，否则按空白分隔
    - 支持 (N,) 或 (N,2) 形式，后者取第二列为计数
    """
    local_path = to_local(path_str)

    with open(local_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else None

    data = np.loadtxt(local_path, dtype=np.float32, delimiter=delimiter)
    if data.ndim == 1:
        counts = data
    else:
        counts = data[:, 1]
    return counts.astype(np.float32)

def mean_spec(df: pd.DataFrame):
    """对一批样本取平均谱；如果 df 为空返回 None。"""
    if df.empty:
        return None
    specs = [load_spec(p) for p in df["file_path"]]
    return np.stack(specs, axis=0).mean(axis=0)

def get_templates(measure_time: float):
    """给定测量时长，取该时长下的典型粘土/粉土平均谱。"""
    clay_ok = pred[
        (pred["true_name"] == "粘土")
        & (pred["true_name"] == pred["pred_name"])
        & (pred["measure_time"] == measure_time)
    ].head(N_TEMPLATES_PER_CLASS)
    silt_ok = pred[
        (pred["true_name"] == "粉土")
        & (pred["true_name"] == pred["pred_name"])
        & (pred["measure_time"] == measure_time)
    ].head(N_TEMPLATES_PER_CLASS)

    mean_clay = mean_spec(clay_ok)
    mean_silt = mean_spec(silt_ok)
    return mean_clay, mean_silt

def main():
    # 只看 true=粉土 的候选
    hard = cands[cands["true_name"] == "粉土"].copy()

    # 如果已经有人为填了 sample_weight，就只画还没填的那些
    if "sample_weight" in hard.columns:
        hard = hard[hard["sample_weight"].isna()]

    # 按 margin 从大到小排序，让你先看“模型最自信地错”的那些
    hard = hard.sort_values("margin_top1_top2", ascending=False)

    for i, (_, row) in enumerate(hard.iterrows()):
        if i >= MAX_CANDIDATES:
            break

        mt = float(row["measure_time"])
        mean_clay, mean_silt = get_templates(mt)
        cand_spec = load_spec(row["file_path"])

        plt.figure(figsize=(10, 4))
        if mean_clay is not None:
            plt.plot(mean_clay, color="tab:orange", label=f"典型粘土({mt:.0f}s)")
        if mean_silt is not None:
            plt.plot(mean_silt, color="tab:green", label=f"典型粉土({mt:.0f}s)")

        plt.plot(
            cand_spec,
            color="red",
            linewidth=2,
            label=f"候选 idx={int(row['idx'])}: 粉土→{row['pred_name']} ({mt:.0f}s)",
        )

        plt.xlabel("channel")
        plt.ylabel("counts")
        plt.legend()
        plt.tight_layout()

    # 所有 figure 一次性显示
    plt.show()

if __name__ == "__main__":
    main()
