# Gamma Energy Spectrum Label Classification Prediction

820 通道伽马能谱 `.txt` 三分类科研项目，目标类别为：

- 粘土
- 砂土
- 粉土

当前工程已经演进为：

- TriBranch CNN + GradientBoosting + XGBoost 集成
- Phase 2 stacking
- 方案 A：`hierarchical + threshold`
- 方案 B1：仅 ML 分支做层级训练

## Project Layout

- [configs/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs)
  - `config.base.json`: shared base config
  - `config.experiment_b1.json`: current B1 experiment overlay
  - `config.local.json`: local path / batch / worker overlay
  - `config.autodl.json`: 5090 / AutoDL overlay
  - `config.json`: default entry, currently points to `config.local.json`
- [src/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src)
  - `train_ensemble.py`: main training entry
  - `dataset.py`: data scan, augmentation, engineered features, cache
  - `model.py`: TriBranch model
  - `train.py`: training loop
  - `evaluate.py`: validation
  - `artifacts.py`: metrics / confusion / prediction export
- [scripts/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/scripts)
  - `summarize_artifacts.py`: experiment summary helper

## Environment

Recommended Python version: `3.10+`

```powershell
pip install -r requirements.txt
```

## Local Training

Default entry loads [configs/config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json), which currently equals `config.local.json`.

```powershell
python -u src/train_ensemble.py
```

Phase 2 only:

```powershell
python -u src/train_ensemble.py --phase2-only --oof-path experiments/artifacts/oof_cache.npz
```

## 5090 / AutoDL Training

The simplest switch is to make [configs/config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json) extend the AutoDL overlay:

```json
{
  "extends": "config.autodl.json"
}
```

Or edit `train_dir`, `val_dir`, `batch_size`, and `num_workers` directly in the Linux environment.

## Current Experiment Lines

- `flat baseline`: best for `Acc`
- `hierarchical + threshold`: best for `粉土 F1 / Macro-F1`
- Current local/B1 enhanced line:
  - `training.ml_hierarchical_training = true`
  - `training.ml_hierarchical.stage1.resampling.strategy = smote`
  - `training.ml_hierarchical.stage1.resampling.target_ratio = 0.9`
  - `training.ml_hierarchical.stage1.weighting.mode = effective_number`
  - `training.ml_hierarchical.stage2.resampling.strategy = none`
  - `training.ml_hierarchical.stage_features.enabled = true`
  - `stacking.meta_features = proba+uncertainty+stage`
  - `stacking.strategy = hierarchical`
  - `stacking.stage1_decision = threshold`
  - `stacking.silt_threshold = 0.43`

## B1 Advanced Config

If you are preparing B1.1 / B1.2 / B1.3 experiments, keep the new knobs in experiment overlays instead of editing the core training code. The current rule of thumb is:

- knobs that change Phase 1 outputs or OOF cache behavior should go under `training`
- knobs that only affect final Phase 2 decision can stay under `stacking`

The implemented field names are documented in [configs/b1_advanced_config.md](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/b1_advanced_config.md).

Minimal current B1 overlay example:

```json
{
  "extends": "config.experiment_b1.json",
  "training": {
    "ml_hierarchical_training": true,
    "ml_hierarchical": {
      "stage1": {
        "resampling": {
          "strategy": "smote",
          "target_ratio": 0.9,
          "k_neighbors": 5
        },
        "weighting": {
          "mode": "effective_number",
          "beta": 0.999
        }
      },
      "stage2": {
        "resampling": {
          "strategy": "none",
          "target_ratio": 1.0,
          "k_neighbors": 5
        },
        "weighting": {
          "mode": "none",
          "beta": 0.999
        }
      },
      "stage_features": {
        "enabled": true,
        "include_positive_prob": true
      }
    }
  },
  "stacking": {
    "meta_features": "proba+uncertainty+stage",
    "strategy": "hierarchical",
    "stage1_decision": "threshold",
    "silt_threshold": 0.43
  }
}
```

## Common Commands

List latest artifacts:

```powershell
dir experiments\artifacts
```

Inspect a run:

```powershell
type experiments\artifacts\<DIR>\stacking_oof_metrics_ascii.json
type experiments\artifacts\<DIR>\stacking_oof_accuracy_by_measure_time.json
```

Summarize multiple runs:

```powershell
python scripts\summarize_artifacts.py experiments\artifacts\phase2_phase2_full_proba_uncertainty_* --show-confusion --show-time-accuracy
```

## Notes

- Sample-level downweighting reads `experiments/artifacts/noisy_silt_candidates*.csv`
- Paths are normalized across Windows and Linux so local and 5090 runs stay comparable
- Do not edit `src/train_ensemble.py` or `src/dataset.py` just to switch environments; use config overlays instead
