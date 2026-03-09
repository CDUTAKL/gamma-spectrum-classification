import json
import logging
import os
import random
import numpy as np
import torch


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_val_acc = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch: int, metrics: dict, is_best: bool):
        # 将 NumPy 数组转为原生 Python 类型，确保 weights_only 加载兼容
        safe_metrics = {
            "val_loss": float(metrics.get("val_loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
            "per_class_f1": [float(v) for v in metrics.get("per_class_f1", [])],
        }
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": safe_metrics,
        }
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(state, last_path)
        if is_best:
            self.best_val_acc = metrics.get("accuracy", 0.0)
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)

    def load(self, model, optimizer, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"], state["metrics"]
