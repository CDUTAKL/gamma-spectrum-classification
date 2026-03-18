# 伽马能谱土壤分类项目交接文档
> 文档类型：持续交接 / 当前状态快照  
> 仓库地址：`https://github.com/CDUTAKL/gamma-spectrum-classification`  
> 最近重写时间：2026-03-18  
> 当前代码基线：`dd2ca40 fix: suppress plot glyph warnings in artifacts`

## 1. 项目一句话概述
这是一个基于伽马能谱 `.txt` 文件做三分类土壤识别的项目，类别为：

- `0 = 粘土`
- `1 = 砂土`
- `2 = 粉土`

主入口是 [`src/train_ensemble.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py)，主流程是：

- TriBranch CNN 集成
- GradientBoosting
- XGBoost
- Phase 2 stacking / hierarchical stacking

当前真正的难点不是“模型能不能训起来”，而是：

- 如何稳定提升 `粉土` 这一 hardest class 的识别能力
- 如何在不明显牺牲总 `Acc` 的情况下提升 `Macro-F1`


## 2. 当前项目所处阶段
项目已经不在“搭框架”阶段，而在“高平台期精修”阶段。

已经完成的关键基础设施：

- OOF cache
- `Phase2-only`
- artifacts 产物导出
- AutoDL / 5090 训练流程
- 本地 GTX 1650 稳定性修复
- 错误样本人工审查与样本级降权
- hierarchical stacking（方案 A）
- hierarchical Stage 1 阈值决策

当前状态可以概括为：

**flat baseline 已稳定；hierarchical 方向已证明有效；当前最值得继续验证的是 hierarchical + tuned threshold。**


## 3. 数据与评估口径
### 3.1 输入
- 每个样本是一个 `.txt`
- 每个样本包含 `820` 通道计数
- 每个样本带测量时长：`30s / 60s / 120s / 150s`

### 3.2 本地路径
- 训练：`E:/data/xunlian`
- 验证：`E:/data/yanzheng`

### 3.3 AutoDL 路径
- 训练：`/root/autodl-tmp/gamma/gamma_data/xunlian`
- 验证：`/root/autodl-tmp/gamma/gamma_data/yanzheng`

### 3.4 评估口径
完整训练时会合并 `train_dir + val_dir`，使用 `StratifiedKFold` 做 OOF。

固定观察结论：

- `30s` 一直最难
- `60s` 次难
- `120s / 150s` 明显更稳

所以当前真正拖后腿的是：

- 短时长
- 粉土边界样本


## 4. 当前关键文件
- [`src/train_ensemble.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py)
  - 主训练入口
  - Phase 1 / Phase 2
  - flat stacking 与 hierarchical stacking
  - Stage 1 threshold 决策逻辑

- [`configs/config.json`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json)
  - 当前主配置

- [`src/dataset.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/dataset.py)
  - 文件解析
  - 特征提取
  - 样本级权重
  - DataLoader 构建

- [`src/train.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train.py)
  - CNN 单模型训练循环

- [`src/evaluate.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/evaluate.py)
  - 验证指标
  - TensorBoard confusion matrix 绘图

- [`src/artifacts.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/artifacts.py)
  - 完整训练后 metrics / confusion / per-class F1 / misclassified 导出

- [`experiments/artifacts/`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts)
  - 所有完整训练与 `Phase2-only` 产物

- [`plot_noisy_silt.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/plot_noisy_silt.py)
  - 人工审查 hard silt 样本用图脚本
  - 注意：此文件当前本地仍有未提交修改，不要误带入别的提交


## 5. 当前配置快照
当前 [`configs/config.json`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json) 关键项：

- `window_feature_dim = 87`
- `use_transformer_branch = false`
- `batch_size = 32`
- `num_workers = 0`
- `meta_features = proba+uncertainty`
- `meta_learner = logreg`
- `strategy = hierarchical`
- `stage1_decision = threshold`
- `silt_threshold = 0.43`

这意味着：

- 当前本地主线不是 flat，而是 hierarchical
- 当前默认阈值已经不是 `0.50`，而是 `0.43`


## 6. 已做过且已经证伪 / 收益递减的方向
### 6.1 已确认有效，但收益已接近上限
- `round2` 人工审查 + sample_weight 软处理

这是历史上最有价值的一次精修，直接提升了主线质量。

### 6.2 已进入收益递减
- `round3` 再扩 hard silt 样本人工审查

有帮助，但提升已经很小，不建议继续 `round4 / round5`。

### 6.3 已试过但未击中主瓶颈
- 第一批 physics-guided features
- 第一批 robust local peak features

结论：

- 不是完全没信息
- 但没有稳定刷新主线
- 甚至有过“帮到砂土、但伤到粉土”的情况


## 7. flat baseline 与 hierarchical 的当前定位
### 7.1 flat baseline
flat baseline 经过重复实验后，已经测出稳定区间：

- `Acc ≈ 0.772`
- `Macro-F1 ≈ 0.742`
- 粉土 `F1 ≈ 0.596`

它的定位是：

- 作为 `Acc` 导向的稳定对照基线
- 作为最传统、最好解释的参考主线

历史上的 flat 单次高点：

- [`phase2_phase2_full_proba_uncertainty_20260312_224059`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260312_224059)
- `Acc = 0.7762`
- `Macro-F1 = 0.7470`

但这个值目前更像高点，不像稳定均值。

### 7.2 hierarchical（方案 A）
方案 A 的定义是：

- 不改 Phase 1 基模型训练目标
- 只改 Phase 2 判别逻辑
- 第一层：`粉土 vs 非粉土`
- 第二层：`粘土 vs 砂土`

它的定位是：

- 作为 `粉土 F1 / Macro-F1` 导向的升级线
- 当前比继续堆特征更值得继续优化

默认阈值 `0.50` 时，完整训练多轮结果说明：

- 相比 flat，`Acc` 没有明显拉开
- 但粉土 `F1` 与 `Macro-F1` 更好一些


## 8. Stage 1 threshold 调优结论
### 8.1 为什么要调 threshold
hierarchical 把 hardest problem 拆出来了：

- “这个样本到底是不是粉土”

所以最有价值的旋钮不是再加一批特征，而是：

- Stage 1 直接判粉土的阈值 `silt_threshold`

### 8.2 第一轮粗扫
粗扫范围：

- `0.45 / 0.50 / 0.55 / 0.60`

结论：

- `0.45` 明显好于 `0.50+`
- 说明之前默认阈值偏高，压住了粉土

### 8.3 第二轮细扫
细扫范围：

- `0.41 / 0.42 / 0.43 / 0.44`

`Phase2-only + OOF cache` 结果里，当前最优点是：

- `0.43`

对应最优 `Phase2-only` 结果：

- 目录：[`phase2_phase2_only_proba_uncertainty_20260318_195149`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_only_proba_uncertainty_20260318_195149)
- `Acc = 0.7795`
- `Macro-F1 = 0.7536`
- 粉土 `F1 = 0.6247`

为什么选 `0.43`：

- 比 `0.41 / 0.42` 更少把粘土错误吸进粉土
- 比 `0.44` 又多保住了一些粉土
- 在混淆矩阵上最像真正的平衡点

### 8.4 重要提醒
`0.43` 目前是：

- **固定 OOF 下的最优 threshold**

它还不等于：

- **完整训练稳态一定就是 0.7795**

也就是说：

- `Phase2-only` 负责找方向
- 完整训练负责确认这个方向能不能站住


## 9. 当前完整训练结果怎么解读
### 9.1 hierarchical + 0.42 的完整训练
目前已经看到的完整训练结果说明：

- `0.42` 对粉土 `F1 / Macro-F1` 是有帮助的
- 但总 `Acc` 还没有稳定复现 `Phase2-only` 的高点

典型完整训练结果：

- [`phase2_phase2_full_proba_uncertainty_20260318_193134`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260318_193134)
- `Acc = 0.7728`
- `Macro-F1 = 0.7456`
- 粉土 `F1 = 0.6105`

结论：

- hierarchical + tuned threshold 对粉土和 Macro-F1 是正向的
- 但完整训练存在 Phase 1 波动，不能直接拿 `Phase2-only` 最优数值当完整训练稳态

### 9.2 当前最稳妥的双主线写法
如果现在要定稿，最稳妥的口径是：

- `flat baseline`
  - 作为 `Acc` 导向主线 / 对照基线

- `hierarchical + tuned threshold`
  - 作为 `粉土 F1 / Macro-F1` 导向主线

这样写是因为：

- 当前“总 Acc 最优”和“粉土/Macro-F1 最优”还没有完全统一到同一条线上


## 10. 当前代码层面的最近有效改动
### 10.1 训练/数据传输提速
提交：

- `71cee74 perf: improve dataloader and device transfer efficiency`

内容：

- DataLoader 统一通过 helper 构建
- `persistent_workers=True`（仅 `num_workers>0`）
- `prefetch_factor=2`（仅 `num_workers>0`）
- `pin_memory` 仅在 CUDA 环境下启用
- 训练/验证/推理改用 `non_blocking` device transfer
- `optimizer.zero_grad(set_to_none=True)`
- 推理路径改为 `torch.inference_mode()`

这些改动不改变实验逻辑，只减少训练和推理公共开销。

### 10.2 训练后字体 warning 修复
提交：

- `dd2ca40 fix: suppress plot glyph warnings in artifacts`

内容：

- 绘图时自动检查是否有可用中文字体
- 如果没有，则自动退回英文标签
- 解决 AutoDL 上反复出现的：
  - `Glyph missing from font(s) DejaVu Sans`

不影响：

- metrics
- csv
- misclassified
- 训练结果本身


## 11. OOF cache / Phase2-only 仍然是关键基础设施
当前已稳定支持：

- `--oof-path`
- `--phase2-only`
- `--save-oof`
- `--meta-features`

关键文件：

- [`experiments/artifacts/oof_cache.npz`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/oof_cache.npz)

用途：

- 完整训练 Phase 1 一次后保存 OOF
- 后面只改 stacking / hierarchical / threshold 时，直接几秒内重跑 Phase 2

当前阈值 sweep 就是靠这套能力完成的。


## 12. 当前最推荐的下一步
如果继续沿方案 A 深挖，推荐顺序是：

1. 先固定 `silt_threshold = 0.43`
2. 做 1~2 次完整训练验证
3. 重点看：
   - `Macro-F1`
   - 粉土 `F1`
   - 总 `Acc`
4. 再决定 hierarchical + 0.43 是否正式收口

如果目标优先级是：

- `Acc`：保留 flat baseline 为主线
- `粉土 F1 / Macro-F1`：优先 hierarchical + tuned threshold

不建议现在优先做：

- 新一轮大规模人工审查
- 再加一批物理特征
- 直接上方案 B 全链路重构

原因：

- 方案 A 还没有被彻底吃干净
- 而且它已经是当前最有证据支持的优化方向


## 13. 5090 常用操作要点
项目目录：

- `/root/autodl-tmp/gamma/gamma-spectrum-classification`

更新代码常用命令：

```bash
cd /root/autodl-tmp/gamma/gamma-spectrum-classification
git restore configs/config.json
git restore src/train_ensemble.py
git pull
git log -1 --oneline
```

服务器路径改写：

```bash
sed -i 's#"train_dir": "E:/data/xunlian"#"train_dir": "/root/autodl-tmp/gamma/gamma_data/xunlian"#' configs/config.json
sed -i 's#"val_dir": "E:/data/yanzheng"#"val_dir": "/root/autodl-tmp/gamma/gamma_data/yanzheng"#' configs/config.json
sed -E -i 's/"batch_size": *[0-9]+/"batch_size": 512/' configs/config.json
sed -E -i 's/"num_workers": *[0-9]+/"num_workers": 16/' configs/config.json
```

训练命令：

```bash
python -u src/train_ensemble.py
```

训练后“老三样”：

1. `stacking_oof_metrics.json`
2. `stacking_oof_predictions.csv` 的混淆矩阵
3. `measure_time` 分组准确率


## 14. 接手时最短摘要
如果新接手的人只看一段，请看这一段：

- 项目当前已经稳定，不缺基础设施
- flat baseline 的稳态约是 `Acc 0.772 / Macro-F1 0.742 / 粉土F1 0.596`
- hierarchical 方向是当前最值得继续的升级方向
- 方案 A 已做完：只改 Phase 2，不改基模型训练
- 当前最重要的新结论是：
  - `Stage 1 threshold` 是关键旋钮
  - `0.43` 是当前 `Phase2-only` 下的最优阈值
- 当前代码默认已经是：
  - `hierarchical + threshold + 0.43`
- 如果追总 `Acc`，flat 仍是最稳对照主线
- 如果追 `粉土 F1 / Macro-F1`，hierarchical + tuned threshold 是当前最合适的主线
