# Gamma Energy Spectrum Label Classification Prediction

基于 820 通道伽马能谱 `.txt` 文件的三分类科研项目，当前任务是区分：

- 粘土
- 砂土
- 粉土

项目主线已经演进到：

- TriBranch CNN + GradientBoosting + XGBoost 异构集成
- Phase 2 stacking
- 方案 A：`hierarchical + threshold`
- 方案 B1：仅 ML 分支做层级训练

## 项目结构

- [configs/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs)
  - `config.base.json`: 共享基础配置
  - `config.experiment_b1.json`: 当前 B1 实验配置
  - `config.local.json`: 本机路径与 batch/worker 配置
  - `config.autodl.json`: 5090/AutoDL 路径与 batch/worker 配置
  - `config.json`: 默认入口，当前指向 `config.local.json`
- [src/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src)
  - `train_ensemble.py`: 主训练入口
  - `dataset.py`: 数据扫描、增强、工程特征、缓存
  - `model.py`: TriBranch 模型
  - `train.py`: 训练循环
  - `evaluate.py`: 验证
  - `artifacts.py`: 导出 metrics / confusion / predictions
- [scripts/](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/scripts)
  - `summarize_artifacts.py`: 汇总实验结果

## 环境安装

建议使用 Python 3.10+。

```powershell
pip install -r requirements.txt
```

## 本机训练

默认入口读取 [configs/config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json)，当前等价于 `config.local.json`。

```powershell
python -u src/train_ensemble.py
```

如果只跑 Phase 2：

```powershell
python -u src/train_ensemble.py --phase2-only --oof-path experiments/artifacts/oof_cache.npz
```

## 5090 / AutoDL 训练

如果要在 5090 上直接切换到服务器配置，最简单的方式是把 [configs/config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json) 临时改成：

```json
{
  "extends": "config.autodl.json"
}
```

或者在 Linux 上手工改 `train_dir / val_dir / batch_size / num_workers`。

## 当前实验线

- `flat baseline`: Acc 导向对照线
- `hierarchical + threshold`: 粉土 F1 / Macro-F1 导向主线
- 当前本地默认实验：B1 初版
  - `training.ml_hierarchical_training = true`
  - `stacking.strategy = hierarchical`
  - `stacking.stage1_decision = threshold`
  - `stacking.silt_threshold = 0.43`

## 常用结果命令

查看最新 artifacts：

```powershell
dir experiments\artifacts
```

查看某轮 metrics：

```powershell
type experiments\artifacts\<DIR>\stacking_oof_metrics_ascii.json
type experiments\artifacts\<DIR>\stacking_oof_accuracy_by_measure_time.json
```

批量汇总：

```powershell
python scripts\summarize_artifacts.py experiments\artifacts\phase2_phase2_full_proba_uncertainty_* --show-confusion --show-time-accuracy
```

## 备注

- 样本级降权会读取 `experiments/artifacts/noisy_silt_candidates*.csv`
- 当前代码已经做了跨平台路径归一化，本机和 5090 使用同一份候选 CSV 时能保持可比
