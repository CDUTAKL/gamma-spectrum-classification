import json
import os
import subprocess
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _maybe_open(path: str, enabled: bool) -> None:
    """在本机环境中尝试打开文件(如 PNG/CSV)。

    注意:
      - AutoDL/Jupyter 等无 GUI 环境下不建议开启, 否则可能无效果或报错。
      - 失败会静默忽略, 不影响训练主流程。
    """
    if not enabled:
        return
    try:
        abs_path = os.path.abspath(path)
        if os.name == "nt":
            os.startfile(abs_path)  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
            return
        subprocess.Popen(["xdg-open", abs_path])
    except Exception:
        return


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str,
    normalize: bool,
) -> None:
    # matplotlib 仅用于离线保存; 强制使用 Agg 避免无显示环境报错。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_disp = cm.astype(np.float64)
    fmt = "d"
    if normalize:
        row_sum = cm_disp.sum(axis=1, keepdims=True)
        cm_disp = np.divide(
            cm_disp,
            row_sum,
            out=np.zeros_like(cm_disp, dtype=np.float64),
            where=row_sum != 0,
        )
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        _ensure_dir(save_dir)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _plot_class_f1(
    class_names: list[str],
    per_class_f1: list[float],
    title: str,
    save_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(class_names, per_class_f1, color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title(title)
    for i, v in enumerate(per_class_f1):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        _ensure_dir(save_dir)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_stacking_oof_artifacts(
    artifact_dir: str,
    file_paths: Iterable[str],
    measure_times: Iterable[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: list[str],
    meta_fold: Optional[np.ndarray] = None,
    auto_open: bool = False,
    logger=None,
) -> dict:
    """保存 Stacking OOF 的可视化与明细表.

    产出内容(默认):
      - stacking_oof_confusion_counts.png
      - stacking_oof_confusion_norm.png
      - stacking_oof_metrics.json
      - stacking_oof_report.txt
      - stacking_oof_predictions.csv
      - stacking_oof_misclassified.csv
      - stacking_oof_per_class_f1.png
    """
    artifact_dir = _ensure_dir(artifact_dir)

    file_paths = list(file_paths)
    measure_times = list(measure_times)
    n = len(file_paths)
    if len(measure_times) != n:
        raise ValueError("file_paths 与 measure_times 长度不一致")
    if len(y_true) != n or len(y_pred) != n:
        raise ValueError("y_true/y_pred 与 file_paths 长度不一致")
    if meta_fold is not None and len(meta_fold) != n:
        raise ValueError("meta_fold 与 file_paths 长度不一致")

    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    counts_png = os.path.join(artifact_dir, "stacking_oof_confusion_counts.png")
    norm_png = os.path.join(artifact_dir, "stacking_oof_confusion_norm.png")
    _plot_confusion_matrix(
        cm,
        class_names,
        title="Stacking OOF Confusion Matrix (Counts)",
        save_path=counts_png,
        normalize=False,
    )
    _plot_confusion_matrix(
        cm,
        class_names,
        title="Stacking OOF Confusion Matrix (Normalized by True)",
        save_path=norm_png,
        normalize=True,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_json = os.path.join(artifact_dir, "stacking_oof_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    report_txt = os.path.join(artifact_dir, "stacking_oof_report.txt")
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")

    # 逐样本明细
    df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int64),
            "file_path": file_paths,
            "measure_time": np.array(measure_times, dtype=np.float64),
            "true": y_true.astype(np.int64),
            "true_name": [class_names[int(t)] for t in y_true],
            "pred": y_pred.astype(np.int64),
            "pred_name": [class_names[int(p)] for p in y_pred],
            "correct": (y_true == y_pred).astype(np.int8),
        }
    )
    if meta_fold is not None:
        df["meta_fold"] = meta_fold.astype(np.int64)

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.shape != (n, len(class_names)):
            raise ValueError(
                f"y_prob 维度不匹配, 期望 {(n, len(class_names))}, 实际 {y_prob.shape}"
            )
        top1 = y_prob.max(axis=1)
        sort2 = np.sort(y_prob, axis=1)
        top2 = sort2[:, -2] if sort2.shape[1] >= 2 else np.zeros(n, dtype=np.float64)
        margin = top1 - top2
        true_prob = y_prob[np.arange(n), y_true]
        df["prob_top1"] = top1
        df["prob_true"] = true_prob
        df["margin_top1_top2"] = margin

    all_csv = os.path.join(artifact_dir, "stacking_oof_predictions.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    mis_df = df[df["correct"] == 0].copy()
    if "margin_top1_top2" in mis_df.columns:
        mis_df = mis_df.sort_values("margin_top1_top2", ascending=True)
    mis_csv = os.path.join(artifact_dir, "stacking_oof_misclassified.csv")
    mis_df.to_csv(mis_csv, index=False, encoding="utf-8-sig")

    # 每类 F1 图
    per_class_f1 = []
    for name in class_names:
        per_class_f1.append(float(report_dict.get(name, {}).get("f1-score", 0.0)))
    f1_png = os.path.join(artifact_dir, "stacking_oof_per_class_f1.png")
    _plot_class_f1(
        class_names,
        per_class_f1,
        title="Stacking OOF Per-Class F1",
        save_path=f1_png,
    )

    if logger is not None:
        logger.info(f"[Artifacts] 已保存至: {os.path.abspath(artifact_dir)}")
        logger.info(f"[Artifacts] 混淆矩阵: {counts_png}")
        logger.info(f"[Artifacts] 误分类清单: {mis_csv}")

    # Windows 本机可选自动弹出(默认关闭)
    _maybe_open(counts_png, enabled=auto_open)
    _maybe_open(mis_csv, enabled=auto_open)

    return {
        "artifact_dir": artifact_dir,
        "confusion_counts_png": counts_png,
        "confusion_norm_png": norm_png,
        "metrics_json": metrics_json,
        "report_txt": report_txt,
        "predictions_csv": all_csv,
        "misclassified_csv": mis_csv,
        "per_class_f1_png": f1_png,
        "n_samples": n,
        "n_misclassified": int((y_true != y_pred).sum()),
    }

