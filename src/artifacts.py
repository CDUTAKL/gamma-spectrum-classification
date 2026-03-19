import json
import os
import subprocess
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ASCII_CLASS_NAME_MAP = {
    "\u7c98\u571f": "Clay",
    "\u7802\u571f": "Sand",
    "\u7c89\u571f": "Silt",
    "绮樺湡": "Clay",
    "鐮傚湡": "Sand",
    "绮夊湡": "Silt",
}

PLOT_NAME_FALLBACKS = {
    "粘土": "Clay",
    "砂土": "Sand",
    "粉土": "Silt",
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_prob_columns(n_classes: int) -> list[str]:
    # Keep columns ASCII-only for easier downstream scripting.
    return [f"prob_{i}" for i in range(int(n_classes))]


def _ascii_class_name(name: str) -> str:
    return ASCII_CLASS_NAME_MAP.get(str(name), str(name))


def _ascii_metrics_dict(report_dict: dict) -> dict:
    """Build an ASCII-only companion report for console-friendly viewing."""
    ascii_report = {}
    for key, value in report_dict.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            ascii_report[key] = value
        else:
            ascii_report[_ascii_class_name(key)] = value
    return ascii_report


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


def _configure_plot_text(matplotlib_module, class_names: list[str]) -> tuple[list[str], str, str]:
    """Return display labels that are safe for the current font environment."""
    from matplotlib import font_manager

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            matplotlib_module.rcParams["font.family"] = [font_name]
            matplotlib_module.rcParams["axes.unicode_minus"] = False
            return list(class_names), "预测标签", "真实标签"

    matplotlib_module.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib_module.rcParams["axes.unicode_minus"] = False
    safe_names = [PLOT_NAME_FALLBACKS.get(name, name) for name in class_names]
    return safe_names, "Predicted", "True"


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
    display_names, xlabel, ylabel = _configure_plot_text(matplotlib, class_names)

    if normalize:
        cm_disp = cm.astype(np.float64)
        row_sum = cm_disp.sum(axis=1, keepdims=True)
        cm_disp = np.divide(
            cm_disp,
            row_sum,
            out=np.zeros_like(cm_disp, dtype=np.float64),
            where=row_sum != 0,
        )
        fmt = ".2f"
    else:
        # seaborn 的 annot+fmt="d" 要求数值是整数；保持计数为 int。
        cm_disp = cm.astype(np.int64)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    display_names, _, _ = _configure_plot_text(matplotlib, class_names)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(display_names, per_class_f1, color="#4C78A8")
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

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_json = os.path.join(artifact_dir, "stacking_oof_metrics.json")
    with open(metrics_json, "w", encoding="utf-8-sig") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    metrics_ascii_json = os.path.join(artifact_dir, "stacking_oof_metrics_ascii.json")
    with open(metrics_ascii_json, "w", encoding="utf-8-sig") as f:
        json.dump(_ascii_metrics_dict(report_dict), f, ensure_ascii=True, indent=2)

    report_txt = os.path.join(artifact_dir, "stacking_oof_report.txt")
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    with open(report_txt, "w", encoding="utf-8-sig") as f:
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

        # Per-class probabilities for error analysis (e.g. 粉土 vs 粘土 margin).
        prob_cols = _safe_prob_columns(len(class_names))
        for i, col in enumerate(prob_cols):
            df[col] = y_prob[:, i].astype(np.float32)

        top1 = y_prob.max(axis=1)
        sort2 = np.sort(y_prob, axis=1)
        top2 = sort2[:, -2] if sort2.shape[1] >= 2 else np.zeros(n, dtype=np.float64)
        margin = top1 - top2
        true_prob = y_prob[np.arange(n), y_true]
        pred_prob = y_prob[np.arange(n), y_pred]
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-12), axis=1)

        df["prob_top1"] = top1
        df["prob_true"] = true_prob
        df["prob_pred"] = pred_prob
        df["margin_top1_top2"] = margin
        df["entropy"] = entropy.astype(np.float32)

    all_csv = os.path.join(artifact_dir, "stacking_oof_predictions.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    mis_df = df[df["correct"] == 0].copy()
    if "margin_top1_top2" in mis_df.columns:
        mis_df = mis_df.sort_values("margin_top1_top2", ascending=True)
    mis_csv = os.path.join(artifact_dir, "stacking_oof_misclassified.csv")
    mis_df.to_csv(mis_csv, index=False, encoding="utf-8-sig")

    # Persist a warning-free by-time summary so downstream checks do not need
    # DataFrameGroupBy.apply(...) on grouping columns.
    by_time_df = (
        df.groupby("measure_time", sort=True)["correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    by_time_json = os.path.join(
        artifact_dir, "stacking_oof_accuracy_by_measure_time.json"
    )
    with open(by_time_json, "w", encoding="utf-8-sig") as f:
        json.dump(
            [
                {
                    "measure_time": float(row.measure_time),
                    "accuracy": float(row.accuracy),
                }
                for row in by_time_df.itertuples(index=False)
            ],
            f,
            ensure_ascii=True,
            indent=2,
        )
    by_time_csv = os.path.join(
        artifact_dir, "stacking_oof_accuracy_by_measure_time.csv"
    )
    by_time_df.to_csv(by_time_csv, index=False, encoding="utf-8-sig")

    # 混淆矩阵与 F1 图：尽量保存，但绘图失败不应影响 CSV/metrics 产出。
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    counts_png = os.path.join(artifact_dir, "stacking_oof_confusion_counts.png")
    norm_png = os.path.join(artifact_dir, "stacking_oof_confusion_norm.png")
    try:
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
    except Exception as e:
        if logger is not None:
            logger.warning(f"[Artifacts] 绘制混淆矩阵失败: {e}")

    # 每类 F1 图
    per_class_f1 = []
    for name in class_names:
        per_class_f1.append(float(report_dict.get(name, {}).get("f1-score", 0.0)))
    f1_png = os.path.join(artifact_dir, "stacking_oof_per_class_f1.png")
    try:
        _plot_class_f1(
            class_names,
            per_class_f1,
            title="Stacking OOF Per-Class F1",
            save_path=f1_png,
        )
    except Exception as e:
        if logger is not None:
            logger.warning(f"[Artifacts] 绘制 per-class F1 失败: {e}")

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
        "metrics_ascii_json": metrics_ascii_json,
        "report_txt": report_txt,
        "predictions_csv": all_csv,
        "misclassified_csv": mis_csv,
        "accuracy_by_measure_time_json": by_time_json,
        "accuracy_by_measure_time_csv": by_time_csv,
        "per_class_f1_png": f1_png,
        "n_samples": n,
        "n_misclassified": int((y_true != y_pred).sum()),
    }
