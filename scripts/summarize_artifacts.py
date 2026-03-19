from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


AVG_KEYS = {"accuracy", "macro avg", "weighted avg"}
SILT_KEYS = ["\u7c89\u571f", "Silt", "silt"]


def _expand_inputs(raw_inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_inputs:
        matches = [Path(p) for p in glob.glob(raw)]
        if matches:
            out.extend(matches)
            continue
        out.append(Path(raw))
    return out


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _find_silt_key(report: dict) -> str | None:
    for key in report.keys():
        key_str = str(key)
        if key_str in AVG_KEYS:
            continue
        lower = key_str.lower()
        if key_str in SILT_KEYS or "silt" in lower or "\u7c89\u571f" in key_str:
            return key_str
    return None


def _find_metric_path(artifact_dir: Path, filename: str) -> Path | None:
    path = artifact_dir / filename
    return path if path.exists() else None


def _load_metrics(artifact_dir: Path) -> dict:
    metrics_path = _find_metric_path(artifact_dir, "stacking_oof_metrics.json")
    if metrics_path is None:
        raise FileNotFoundError(f"missing stacking_oof_metrics.json in {artifact_dir}")
    return _load_json(metrics_path)


def _load_predictions(artifact_dir: Path) -> pd.DataFrame:
    pred_path = _find_metric_path(artifact_dir, "stacking_oof_predictions.csv")
    if pred_path is None:
        raise FileNotFoundError(
            f"missing stacking_oof_predictions.csv in {artifact_dir}"
        )
    return pd.read_csv(pred_path, encoding="utf-8-sig")


def _load_time_accuracy(artifact_dir: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    cached = _find_metric_path(
        artifact_dir, "stacking_oof_accuracy_by_measure_time.csv"
    )
    if cached is not None:
        return pd.read_csv(cached, encoding="utf-8-sig")

    if "measure_time" not in predictions.columns or "correct" not in predictions.columns:
        raise KeyError("predictions must contain measure_time and correct columns")

    return (
        predictions.groupby("measure_time", sort=True)["correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )


def _format_float(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_pct(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def summarize_artifact(artifact_dir: Path) -> dict:
    report = _load_metrics(artifact_dir)
    silt_key = _find_silt_key(report)
    silt_report = report.get(silt_key, {}) if silt_key is not None else {}

    return {
        "artifact_dir": artifact_dir.name,
        "Acc": report.get("accuracy"),
        "Macro-F1": report.get("macro avg", {}).get("f1-score"),
        "silt_precision": silt_report.get("precision"),
        "silt_recall": silt_report.get("recall"),
        "silt_f1": silt_report.get("f1-score"),
    }


def print_summary_table(rows: list[dict]) -> None:
    if not rows:
        print("No artifacts found.")
        return

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col == "artifact_dir":
            continue
        df[col] = df[col].map(_format_float)
    print(df.to_string(index=False))


def summarize_confusion_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(predictions["true_name"], predictions["pred_name"])
    row_total = matrix.sum(axis=1)
    row_acc = []
    for idx in matrix.index:
        total = float(row_total.loc[idx])
        correct = float(matrix.loc[idx, idx]) if idx in matrix.columns else 0.0
        row_acc.append(correct / total if total > 0 else 0.0)

    summary = matrix.copy()
    summary["row_total"] = row_total
    summary["row_acc"] = row_acc
    return summary.reset_index(names="true_name")


def summarize_time_accuracy_wide(time_accuracy: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, object] = {}
    for _, item in time_accuracy.sort_values("measure_time").iterrows():
        mt = item["measure_time"]
        row[f"mt_{int(mt)}"] = item["accuracy"]
    return pd.DataFrame([row])


def print_confusion_matrix(artifact_dir: Path, predictions: pd.DataFrame) -> None:
    print(f"\n[{artifact_dir.name}] confusion matrix")
    matrix = summarize_confusion_matrix(predictions)
    display = matrix.copy()
    for col in display.columns:
        if col == "true_name":
            continue
        if col == "row_acc":
            display[col] = display[col].map(_format_pct)
        else:
            display[col] = display[col].map(_format_float)
    print(display.to_string(index=False))


def print_time_accuracy(artifact_dir: Path, time_accuracy: pd.DataFrame) -> None:
    print(f"\n[{artifact_dir.name}] measure_time accuracy")
    wide = summarize_time_accuracy_wide(time_accuracy)
    for col in wide.columns:
        wide[col] = wide[col].map(_format_pct)
    print(wide.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize artifact metrics across one or more experiment directories."
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="Artifact directories or glob patterns.",
    )
    parser.add_argument(
        "--show-confusion",
        action="store_true",
        help="Also print confusion matrices for each artifact directory.",
    )
    parser.add_argument(
        "--show-time-accuracy",
        action="store_true",
        help="Also print measure_time accuracy summaries.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for the summary table.",
    )
    args = parser.parse_args()

    artifact_dirs = [
        path
        for path in _expand_inputs(args.artifacts)
        if path.is_dir()
    ]
    artifact_dirs.sort(key=lambda p: p.name)

    rows: list[dict] = []
    for artifact_dir in artifact_dirs:
        try:
            rows.append(summarize_artifact(artifact_dir))
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

    print_summary_table(rows)

    if args.csv_out is not None and rows:
        df = pd.DataFrame(rows)
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")
        print(f"\nWrote summary CSV to {args.csv_out}")

    if args.show_confusion or args.show_time_accuracy:
        for artifact_dir in artifact_dirs:
            try:
                predictions = _load_predictions(artifact_dir)
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue

            if args.show_confusion:
                print_confusion_matrix(artifact_dir, predictions)

            if args.show_time_accuracy:
                try:
                    time_accuracy = _load_time_accuracy(artifact_dir, predictions)
                except KeyError as exc:
                    print(f"[skip] {artifact_dir.name}: {exc}")
                else:
                    print_time_accuracy(artifact_dir, time_accuracy)


if __name__ == "__main__":
    main()


