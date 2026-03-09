"""
消融实验：用传统机器学习方法（RandomForest / XGBoost）在工程特征上做 5-Fold CV。
与深度学习模型对比，判断特征质量和模型选择的影响。

用法: python src/train_ml.py
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    scan_directory, load_spectrum, extract_engineered_features,
    fit_pca_statistics,
)
from src.utils import load_config

CLASS_NAMES = ["粘土", "砂土", "粉土"]


def load_all_features(config: dict):
    """加载所有样本的 CPS 谱，供每个 Fold 内部提取完整工程特征。"""
    spectrum_length = config["data"]["spectrum_length"]

    all_cps = []
    all_mts = []
    all_y = []

    for data_dir in [config["data"]["train_dir"], config["data"]["val_dir"]]:
        fps, lbs, mts, skipped = scan_directory(data_dir)
        print(f"  {data_dir}: {len(fps)} 样本, 跳过 {skipped}")
        for fp, lb, mt in zip(fps, lbs, mts):
            raw = load_spectrum(fp, spectrum_length)
            cps = raw / mt
            all_cps.append(cps)
            all_mts.append(mt)
            all_y.append(lb)

    X = np.array(all_cps, dtype=np.float32)
    mts = np.array(all_mts, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"  总样本: {len(y)}, 谱长度: {X.shape[1]}")
    print("  最终工程特征将在每个 Fold 内单独拟合 PCA 后生成")
    for c in range(config["data"]["num_classes"]):
        print(f"  {CLASS_NAMES[c]}: {(y == c).sum()}")
    return X, mts, y


def evaluate_model(name, model_cls, model_params, cps_spectra,
                   measure_times, energy_windows, y, n_splits=5, seed=42):
    """用 StratifiedKFold 评估一个模型，打印每个 Fold 和平均结果。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs = []
    fold_f1s = []

    print(f"\n{'=' * 50}")
    print(f"模型: {name}")
    print(f"{'=' * 50}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cps_spectra, y)):
        fold = fold_idx + 1

        train_cps = cps_spectra[train_idx]
        val_cps = cps_spectra[val_idx]
        train_mts = measure_times[train_idx]
        val_mts = measure_times[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 仅用训练 Fold 拟合 PCA，再分别提取训练/验证工程特征，避免泄漏
        feature_stats = fit_pca_statistics(train_cps)
        X_train = np.array([
            extract_engineered_features(cps, energy_windows, mt, feature_stats)
            for cps, mt in zip(train_cps, train_mts)
        ], dtype=np.float32)
        X_val = np.array([
            extract_engineered_features(cps, energy_windows, mt, feature_stats)
            for cps, mt in zip(val_cps, val_mts)
        ], dtype=np.float32)

        # Z-score 标准化（用训练集统计量）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = model_cls(**model_params, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)

        print(f"  Fold {fold}: Acc={acc:.4f}  F1={f1:.4f}")

        # 最后一个 fold 打印详细报告
        if fold == n_splits:
            print(classification_report(
                y_val, y_pred, target_names=CLASS_NAMES, zero_division=0
            ))

    avg_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    avg_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"  平均 Acc: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  平均 F1:  {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"{'=' * 50}")

    return {"name": name, "avg_acc": avg_acc, "std_acc": std_acc,
            "avg_f1": avg_f1, "std_f1": std_f1}


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config.json"
    )
    config = load_config(config_path)
    seed = config["training"]["seed"]
    n_splits = config["training"].get("n_splits", 5)
    energy_windows = config["data"]["energy_windows"]

    print("加载数据和提取特征...")
    X, mts, y = load_all_features(config)

    # 定义要测试的模型
    models = [
        ("RandomForest (100 trees)", RandomForestClassifier, {
            "n_estimators": 100, "max_depth": None, "min_samples_leaf": 2,
            "class_weight": "balanced",
        }),
        ("RandomForest (300 trees)", RandomForestClassifier, {
            "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2,
            "class_weight": "balanced",
        }),
        ("GradientBoosting (100)", GradientBoostingClassifier, {
            "n_estimators": 100, "learning_rate": 0.1, "max_depth": 4,
            "min_samples_leaf": 5, "subsample": 0.8,
        }),
        ("GradientBoosting (300)", GradientBoostingClassifier, {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
            "min_samples_leaf": 5, "subsample": 0.8,
        }),
    ]

    # 尝试导入 XGBoost
    try:
        from xgboost import XGBClassifier
        models.extend([
            ("XGBoost (100)", XGBClassifier, {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 4,
                "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
                "use_label_encoder": False, "eval_metric": "mlogloss",
            }),
            ("XGBoost (300)", XGBClassifier, {
                "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
                "use_label_encoder": False, "eval_metric": "mlogloss",
            }),
        ])
    except ImportError:
        print("\n[提示] 未安装 xgboost，跳过 XGBoost 实验。安装: pip install xgboost")

    results = []
    for name, model_cls, params in models:
        r = evaluate_model(
            name, model_cls, params, X, mts, energy_windows, y, n_splits, seed
        )
        results.append(r)

    # 汇总对比
    print(f"\n{'=' * 60}")
    print("=== 传统 ML 消融实验汇总 ===")
    print(f"{'=' * 60}")
    print(f"{'模型':<30} {'Acc':>12} {'F1':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30} {r['avg_acc']:.4f}±{r['std_acc']:.4f}  "
              f"{r['avg_f1']:.4f}±{r['std_f1']:.4f}")
    # 加入 DualBranch CNN+MLP 的参考值
    print(f"{'DualBranch CNN+MLP (参考)':<30} {'0.7267±0.0149':>12} {'0.6918±0.0179':>12}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
