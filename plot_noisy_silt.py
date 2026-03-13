import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = "phase2_phase2_full_proba_uncertainty_20260312_224059"
DEFAULT_CANDIDATE_CSV = "noisy_silt_candidates_round2.csv"
DATA_ROOT = Path(r"E:/data")
N_TEMPLATES_PER_CLASS = 20
CLASS_LABELS = {0: "clay", 1: "sand", 2: "silt"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot suspicious silt samples against mean clay/silt templates."
    )
    parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Artifact subdirectory containing stacking_oof_predictions.csv",
    )
    parser.add_argument(
        "--candidate-csv",
        default=DEFAULT_CANDIDATE_CSV,
        help="CSV filename under experiments/artifacts",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum number of candidates to plot",
    )
    return parser.parse_args()


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def to_local(path_str: str) -> Path:
    normalized = str(path_str).replace("\\", "/")
    if "gamma_data/" in normalized:
        rel = normalized.split("gamma_data/", 1)[1]
        return DATA_ROOT / rel
    return Path(path_str)


def load_spec(path_str: str) -> np.ndarray:
    local_path = to_local(path_str)
    with open(local_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else None
    data = np.loadtxt(local_path, dtype=np.float32, delimiter=delimiter)
    if data.ndim == 1:
        return data.astype(np.float32)
    return data[:, 1].astype(np.float32)


def mean_spec(df: pd.DataFrame):
    if df.empty:
        return None
    specs = [load_spec(p) for p in df["file_path"]]
    return np.stack(specs, axis=0).mean(axis=0)


def get_templates(pred: pd.DataFrame, measure_time: float):
    clay_ok = pred[
        (pred["true"] == 0)
        & (pred["pred"] == 0)
        & (pred["measure_time"] == measure_time)
    ].head(N_TEMPLATES_PER_CLASS)
    silt_ok = pred[
        (pred["true"] == 2)
        & (pred["pred"] == 2)
        & (pred["measure_time"] == measure_time)
    ].head(N_TEMPLATES_PER_CLASS)
    return mean_spec(clay_ok), mean_spec(silt_ok)


def filter_candidates(cands: pd.DataFrame) -> pd.DataFrame:
    if "true" in cands.columns:
        hard = cands[cands["true"] == 2].copy()
    else:
        hard = cands[cands["true_name"] == "粉土"].copy()

    if "sample_weight" in hard.columns:
        hard = hard[hard["sample_weight"].isna()]

    return hard.sort_values("margin_top1_top2", ascending=False)


def main():
    args = parse_args()
    proj_root = Path(__file__).resolve().parent
    artifacts_dir = proj_root / "experiments" / "artifacts"
    pred_dir = artifacts_dir / args.artifact_dir
    candidate_path = artifacts_dir / args.candidate_csv

    pred = pd.read_csv(pred_dir / "stacking_oof_predictions.csv", encoding="utf-8")
    cands = read_csv_fallback(candidate_path)
    hard = filter_candidates(cands)

    for i, (_, row) in enumerate(hard.iterrows()):
        if i >= args.max_candidates:
            break

        mt = float(row["measure_time"])
        mean_clay, mean_silt = get_templates(pred, mt)
        cand_spec = load_spec(row["file_path"])
        pred_name = row.get("pred_name", CLASS_LABELS.get(int(row.get("pred", -1)), "unknown"))

        plt.figure(figsize=(10, 4))
        if mean_clay is not None:
            plt.plot(mean_clay, color="tab:orange", label=f"mean clay ({mt:.0f}s)")
        if mean_silt is not None:
            plt.plot(mean_silt, color="tab:green", label=f"mean silt ({mt:.0f}s)")
        plt.plot(
            cand_spec,
            color="red",
            linewidth=2,
            label=f"candidate idx={int(row['idx'])}: silt->{pred_name} ({mt:.0f}s)",
        )
        plt.xlabel("channel")
        plt.ylabel("counts")
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
