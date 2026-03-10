# 伽马能谱土壤分类项目说明报告

> 文档类型: 项目说明报告 (由原“交接文档”升级而来)
>
> 首次成稿: 2026-03-09 (Claude)
>
> 最近更新: 2026-03-10 (Codex)
>
> 代码仓库: `https://github.com/CDUTAKL/gamma-spectrum-classification`

本项目使用 820 通道伽马能谱计数数据，对土壤类型进行三分类(粘土/砂土/粉土)。当前方案以“特征工程 + 三分支深度模型 + 传统 ML + Stacking”为主线，目标是将 Stacking 全局准确率提升到 80%+。

本报告面向“下一位开发者/复现实验者”，核心内容是: 数据与标签约定、特征流水线、模型与训练流程、结果复盘、已知问题与运维要点、关键文件速查。

---

## 一、项目概述

### 1.1 任务定义

输入: 每个样本为一个 `.txt` 文件，包含 820 个通道的计数值(原始 counts)。每个样本对应一个测量时长(30/60/120/150 秒)以及一个土壤类别标签。

输出: 3 类土壤标签：

| 类别 ID | 类别名 | 备注 |
|---|---|---|
| 0 | 粘土 | 与粉土易混 |
| 1 | 砂土 | 相对最容易 |
| 2 | 粉土 | 核心瓶颈(样本少、与粘土特征重叠) |

### 1.2 数据格式与目录

- 输入文件: `.txt`，每行一个通道计数，共 820 行。
- 文件名包含两个关键信息:
  - 测量时长: 例如 `30s`、`60s`、`120s`、`150s`
  - 标签字段: 例如 `标签一/标签二/标签三/标签四` 等(见 `src/dataset.py:parse_filename`)
- 目录(本地 Windows 默认):
  - 训练数据: `E:/data/xunlian`
  - 验证数据: `E:/data/yanzheng`
- 实际训练使用方式:
  - `train_ensemble.py` 会扫描 `train_dir` 与 `val_dir` 两个目录后合并，做 5-Fold 交叉验证(用于更稳定的评估与对比)。

### 1.3 复现一致性要求(用于可比性)

为了让不同实验之间“可比”，需要保持以下三点一致：

1. 数据路径与数据内容不变。
2. 固定随机种子 `seed=42`。
3. 5-Fold 划分方式不变：`StratifiedKFold` + “时间分层”(按测量时长分 bin 与标签联合分层)。

### 1.4 运行环境与依赖

本地推荐环境(Windows)：

- Python: conda 环境 `xsypytorch` (用户自建)
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- GPU: GTX 1650(4GB) 可跑通，但耗时较长

依赖(见 [requirements.txt](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/requirements.txt))：

- `torch/torchvision`, `numpy/scipy/pandas`
- `scikit-learn`, `matplotlib/seaborn`, `tensorboard`, `tqdm`
- `PyWavelets`
- `xgboost` (部分云镜像需要单独 `pip install xgboost`)

---

## 二、仓库结构与产物目录

项目根目录结构(关键项)：

```
Gamma Energy Spectrum Label Classification Prediction/
  configs/
    config.json                 # 训练与特征/模型超参配置
  src/
    dataset.py                  # 数据加载、增强、特征工程、PCA/统计量(重点)
    model.py                    # TriBranch 模型定义、SMAE 等
    train.py                    # 单模型训练循环(K-Fold 内部用)
    train_ensemble.py           # 主入口：集成训练系统 v2
    train_ml.py                 # 传统 ML 训练(备用/辅助)
    pretrain_smae.py            # SMAE 自监督预训练(已提供权重)
    evaluate.py                 # 评估/混淆矩阵绘图(训练中用)
    artifacts.py                # 训练结束导出可视化与逐样本预测明细(新增)
    utils.py                    # logger/config/seed 等
  experiments/
    checkpoints/
      smae_pretrained.pth       # SMAE 预训练权重(已纳入 Git 追踪)
      best_model.pth            # 训练中写入
      last_model.pth            # 训练中写入
    logs/
      train_ensemble_v2.log     # 主训练日志
      events.out.tfevents...    # TensorBoard
    artifacts/                  # 训练结束自动产物(新增)
      stacking_oof_*.png/csv    # 混淆矩阵/正确错误清单等
```

说明：

- `experiments/` 目录通常体积很大，仓库 `.gitignore` 默认忽略，但已“白名单”保留 `experiments/checkpoints/smae_pretrained.pth`，确保云端 `git clone` 后可直接训练。
- 训练结束的可视化/清单输出默认写入 `experiments/artifacts/`，可通过配置修改。

---

## 三、数据处理与特征工程流水线

### 3.1 总流程(单样本)

从文件到模型输入的流程如下：

1. 读取 counts: `load_spectrum(fp) -> raw_counts (820,)`
2. 训练时增强(TTA/训练增强)：
   - Poisson 重采样(模拟统计涨落)
   - 通道平移(模拟能量刻度漂移)
   - 高斯噪声
3. 转 CPS 谱: `cps = raw_counts / measure_time`
4. 三通道输入构造：
   - 通道 1: CPS
   - 通道 2: CPS 的一阶导数(使用 SG 滤波导数)
   - 通道 3: CPS 的二阶导数(使用 SG 滤波导数)
5. 能窗工程特征提取(重点)：`extract_engineered_features(cps, energy_windows, measure_time, stats)`
6. 特征标准化：Z-score(仅使用训练 fold 统计量，避免泄漏)

最终每个样本返回：

- `spectrum`: shape `(3, 820)`
- `window_features`: shape `(D,)`，当前 `D=86`(见 3.4)
- `label`: `int` in `{0,1,2}`

### 3.2 能窗定义

配置在 `config.json:data.energy_windows`：

- K: `[327, 396]`
- U: `[426, 496]`
- Th: `[682, 820]`

这些窗口来自经验设定，用于提取核素相关峰区与背景信息。

### 3.3 统计量与“无泄漏”原则

本项目对特征做了“训练 fold 统计量”驱动的处理，避免验证集信息泄漏到训练中：

- `GammaSpectrumDataset` 会在构建时对训练 fold 做 `_precompute_statistics()`，统计：
  - `feature_mean/feature_std`：用于窗口特征 Z-score
  - `pca`：在训练 fold 的“基础特征”上拟合 PCA，再对训练/验证 fold 做 transform
- 传统 ML 分支也会使用相同的 `stats`(由 `extract_ml_features(..., feature_stats=stats)` 注入)，保证 ML 的 PCA 特征同样“无泄漏”。

### 3.4 当前窗口特征维度 D=86 (含已落地优化 1-5)

`window_feature_dim` 位于 `config.json:model.window_feature_dim`，当前为 `86`。其组成如下(以实现为准)：

1. 基础能窗/物理先验特征：`71` 维
2. PCA 得分特征：`15` 维

合计: `71 + 15 = 86`

其中“已落地的优化项(1-5)”对应的关键增量：

- 优化 1: SG 滤波导数(影响谱输入 3 通道的质量，不改变 D)
- 优化 2: 小波能量特征(并入窗口特征，增加若干维)
- 优化 3: PCA 得分特征(+15 维，训练 fold 拟合)
- 优化 4: 对数比值特征(增加若干维)
- 优化 5: 峰局部对比/局部矩特征(每个能窗 4 维，共 12 维)

#### 3.4.1 峰局部对比特征(优化 5)定义(实现口径)

对每个能窗(K/U/Th)提取 4 维，代码在 `dataset.py:_extract_peak_local_contrast_features`：

- `peak_ratio`: 峰高 / 局部背景均值
- `peak_snr`: (峰高 - 背景) / 背景噪声(MAD 近似)
- `peak_offset`: 峰位置相对能窗中心的偏移(归一化到 [0,1] 左右)
- `peak_focus`: 峰能量集中度(局部能量占比的 proxy)

这些特征的目标是增强“峰显著性、峰定位可靠性、峰形差异”的表征，针对粉土/粘土易混问题做补强。

---

## 四、模型架构与训练系统

### 4.1 TriBranchModel(三分支融合)

TriBranchModel 输入为两路：

- `spectrum`: `(B, 3, 820)`，三通道谱(原谱 + 一阶导 + 二阶导)
- `window_features`: `(B, D)`，工程特征(D=86)

模型包含三条分支并融合：

1. Multi-Scale SE-CNN 分支：多尺度卷积核 + SE 注意力，擅长捕获局部峰形/峰宽等局部特征。
2. Spectral Transformer 分支：对谱段做 patch embedding 后进入 TransformerEncoder，捕获长程关联与跨谱段模式。
3. MLP 先验分支：直接编码窗口工程特征，补充物理先验、比例关系等可解释信息。

实现位置: `src/model.py`。

补充说明：

- `nn.TransformerEncoder(enable_nested_tensor=False)`：显式关闭 nested tensor 优化以消除 PyTorch 的兼容性 warning(在 `norm_first=True` 时常见)。
- AMP 混合精度训练已升级到 `torch.amp.*` 新接口(避免 FutureWarning)。

### 4.2 集成训练系统(train_ensemble.py)

`src/train_ensemble.py` 是主入口，采用两阶段：

Phase 1: 5-Fold 训练基模型，收集 out-of-fold(OoF) 概率

- 深度模型：TriBranch CNN 集成(3 个 seed：42/43/44)
  - 每 fold 内训练 3 个模型
  - 用 TTA(n=3) 对验证折预测并平均，得到 `oof_cnn (N,3)`
- 传统 ML 基模型：
  - GradientBoosting + SMOTE，得到 `oof_gb (N,3)`
  - XGBoost + SMOTE，得到 `oof_xgb (N,3)`

Phase 2: Stacking 元学习器评估

- 拼接元特征：`meta_X = [oof_cnn, oof_gb, oof_xgb] -> (N, 9)`
- 使用“独立的 StratifiedKFold 划分”(不同 random_state)做元学习器 CV，降低间接信息通路带来的偏乐观评估风险。
- 元学习器：XGBoost 分类器

### 4.3 torch.compile 策略(Windows 兼容优先)

由于 Windows 下 `inductor/triton` 后端存在“临时目录清理阶段 WinError 5”等不稳定情况，项目目前将编译后端固定为：

```python
model = torch.compile(model, backend="aot_eager")
```

特点：

- 可避免 triton kernel 编译阶段，从而绕开 WinError 5。
- 有一定加速(通常小于 inductor)，但优先保证可跑通与稳定。

涉及文件：

- `src/train.py`
- `src/train_ensemble.py`
- `src/pretrain_smae.py`

---

## 五、训练评估、日志与可视化产物

### 5.1 训练中可视化(TensorBoard + 训练期混淆矩阵)

- `evaluate.py` 会在验证时计算混淆矩阵，并保存 `experiments/logs/confusion_matrix_epochXXX.png`。
- TensorBoard 日志写入 `experiments/logs/`，可用：

```bash
tensorboard --logdir experiments/logs
```

### 5.2 训练结束自动产物(新增)

训练完成后(Phase 2 Stacking 全局指标输出后)，会自动导出以下文件到：

- 默认目录：`experiments/artifacts/`
- 可配置：`config.json:output.artifact_dir`

产物清单(文件名固定，便于脚本化收集)：

- `stacking_oof_confusion_counts.png`：OOF 混淆矩阵(计数)
- `stacking_oof_confusion_norm.png`：OOF 混淆矩阵(按真实类别归一化)
- `stacking_oof_per_class_f1.png`：每类 F1 柱状图
- `stacking_oof_report.txt`：classification_report 文本版
- `stacking_oof_metrics.json`：classification_report 的结构化 JSON
- `stacking_oof_predictions.csv`：每个样本的预测明细(含 `correct` 列，标记预测成功/失败)
- `stacking_oof_misclassified.csv`：仅误分类样本清单(默认按 margin 从小到大排序，便于定位“最不确定”的错误样本)

Windows 可选“自动弹出”(默认关闭)：

- `config.json:output.auto_open_artifacts = true`
- 会尝试自动打开混淆矩阵 PNG 与误分类 CSV
- AutoDL/无 GUI 环境建议保持 `false`

对应实现：`src/artifacts.py`。

---

## 六、训练结果汇总(截至 2026-03-10)

### 6.1 历史对比(以 Stacking 为准)

文档早期记录的 Run 1-3 为“旧特征版本”下的结果(见原文档历史)，核心基线是：

- Run 3(2026-03-09)：Stacking 全局 `Acc=0.7613`，Macro-F1 `0.7279`

在落地“优化 1-5(特征工程 + 峰局部对比)”后，最近一次完成训练记录为：

- Run 4(2026-03-10)：Stacking 全局 `Acc=0.7661`，Macro-F1 `0.7313`

结论(现阶段)：特征工程方向是正收益，但离 80% 仍有距离，“粉土”仍是主要瓶颈。

### 6.2 当前痛点复盘

- 粉土样本数量较少，且与粘土在 K/U/Th 相关比例上重叠明显，导致混淆高。
- Fold 间差异显著，部分 fold 的验证集包含更多边界样本，这是正常现象，不建议为单一 fold 特化。

---

## 七、深度分析与后续优化建议(概念层面)

本节不再作为“交接任务清单”，而作为“可选路线图”。已落地项为 1-5。

建议下一阶段仍遵循“先低风险特征工程，再动模型”的原则：

- 优先继续完善特征工程的可解释性与鲁棒性：
  - 更精细的峰背景估计(局部拟合/更稳健噪声估计)
  - 结合能窗的物理约束做一致性特征(跨窗比值、对数域约束)
- 若特征工程接近瓶颈，再考虑中等风险改动：
  - 数据增强：Manifold Mixup、1D CutMix
  - 架构微调：跨窗口交叉注意力
  - 预训练策略：SupCon 监督对比预训练

---

## 八、实施注意事项与常见问题排查

### 8.1 Windows 运行注意事项

- `python.exe src/train_ensemble.py` 报 `No module named 'torch'`：
  - 说明当前终端的 `python.exe` 不是 conda 环境里的 Python。
  - 用 `where python` 确认路径；或先 `conda activate xsypytorch` 再运行 `python src/train_ensemble.py`。
- `num_workers > 0` 在 Windows 上可能卡住或不稳定：保持 `config.json:training.num_workers=0`。

### 8.2 torch.compile 相关

- Windows 下 `inductor/triton` 后端不稳定时，优先使用 `backend="aot_eager"`(当前已固定)。
- 如果仍遇到编译阶段异常，可临时将 `config.json:training.use_compile=false`，以稳定训练为第一目标。

### 8.3 XGBoost 相关

- 云端镜像缺少 `xgboost` 时，会在导入阶段报错：
  - `pip install xgboost`

---

## 九、如何运行(本地/云端)

### 9.1 本地 Windows(推荐最简流程)

1. 激活环境：

```bash
conda activate xsypytorch
```

2. 运行训练：

```bash
python -u src/train_ensemble.py
```

3. 查看日志：

- 主日志：`experiments/logs/train_ensemble_v2.log`
- 训练结束产物：`experiments/artifacts/`

### 9.2 AutoDL/云端(建议用于加速迭代)

1. 租用 GPU 节点后，在终端执行：

```bash
cd /root
git clone https://github.com/CDUTAKL/gamma-spectrum-classification.git
cd gamma-spectrum-classification
```

2. 安装依赖：

```bash
pip install -r requirements.txt
pip install xgboost
```

3. 上传数据到云端目录(建议直接上传文件夹而非 zip)：

- 目标结构建议为：
  - `/root/gamma-spectrum-classification/data/xunlian`
  - `/root/gamma-spectrum-classification/data/yanzheng`

4. 修改 `configs/config.json` 的数据路径：

```json
{
  "data": {
    "train_dir": "/root/gamma-spectrum-classification/data/xunlian",
    "val_dir": "/root/gamma-spectrum-classification/data/yanzheng"
  }
}
```

5. 后台启动训练并写日志：

```bash
nohup python -u src/train_ensemble.py > train_output.log 2>&1 &
tail -f train_output.log
```

6. 训练完成后下载结果：

- 产物目录：`experiments/artifacts/`
- 也可下载 `experiments/logs/` 用于 TensorBoard 回放

---

## 十、关键文件速查

| 需求 | 文件 | 说明 |
|---|---|---|
| 数据解析/特征工程主逻辑 | [src/dataset.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/dataset.py) | SG 导数、对数比值、小波能量、PCA、峰局部对比等均在这里 |
| 集成训练主入口 | [src/train_ensemble.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py) | Phase1 OoF + Phase2 stacking；训练结束导出 artifacts |
| 单模型训练循环 | [src/train.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train.py) | AMP、scheduler、早停、SWA 等 |
| 模型结构 | [src/model.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/model.py) | TriBranch、Transformer(enable_nested_tensor=False) |
| 训练期评估/混淆矩阵 | [src/evaluate.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/evaluate.py) | 每 epoch 评估与混淆矩阵绘图 |
| 训练结束可视化/清单导出 | [src/artifacts.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/artifacts.py) | 导出“预测成功/失败”清单与图表 |
| 全局配置 | [configs/config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json) | `window_feature_dim=86`、artifact_dir 等 |

