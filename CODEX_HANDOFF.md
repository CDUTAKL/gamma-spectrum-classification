# 伽马能谱土壤分类项目说明与交接文档

> 文档类型：项目现状说明 / 持续交接文档
>
> 仓库地址：`https://github.com/CDUTAKL/gamma-spectrum-classification`
>
> 最近重写时间：2026-03-13
>
> 重写时本地分支提交：`352a3d8 feat: add physics-guided silt features`

## 一、项目当前处于什么阶段

这个项目已经不再是早期“搭框架”阶段，而是进入了：

- 主流程稳定可训练
- 已做过多轮完整实验
- 已完成误分类驱动的人工审查与软处理
- 当前主要矛盾聚焦在“粉土”与“粘土/砂土”的边界样本
- 后续优化重点应转向更贴近顽固粉土错分的特征设计

一句话概括当前状态：

**模型框架已经成型，当前瓶颈不是能不能训练，而是如何把 persistent 粉土错分进一步压下去。**


## 二、任务定义

输入：
- 每个样本为一个 `.txt` 文件
- 含 820 个通道的伽马能谱计数
- 每个样本带测量时长：`30s / 60s / 120s / 150s`

输出：
- 三分类土壤标签
  - `0`: 粘土
  - `1`: 砂土
  - `2`: 粉土

目标：
- `Stacking Acc >= 0.80`

当前实际情况：
- 最强结果仍在 `0.776` 左右
- 粉土仍是主瓶颈


## 三、当前主入口与核心文件

主入口：
- [`src/train_ensemble.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py)

关键文件：

- [`configs/config.json`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json)
  - 训练、模型、特征、stacking 配置

- [`src/dataset.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/dataset.py)
  - 文件解析
  - 数据增强
  - 工程特征提取
  - PCA 统计
  - `sample_weight` 样本级降权逻辑

- [`src/model.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/model.py)
  - TriBranch 模型定义
  - transformer 分支可通过配置关闭

- [`src/train.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train.py)
  - 单模型训练工具函数

- [`src/artifacts.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/artifacts.py)
  - 训练后产物导出
  - 混淆矩阵、F1 图、预测明细、误分类清单

- [`plot_noisy_silt.py`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/plot_noisy_silt.py)
  - 本地人工审查 hard silt 样本时的辅助出图脚本
  - 注意：这个文件当前本地还有未提交修改

- [`experiments/artifacts/`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts)
  - 完整训练产物
  - OOF cache
  - 人工审查 CSV


## 四、数据路径与评估口径

本地默认路径：
- `E:/data/xunlian`
- `E:/data/yanzheng`

AutoDL 路径：
- `/root/autodl-tmp/gamma/gamma_data/xunlian`
- `/root/autodl-tmp/gamma/gamma_data/yanzheng`

评估规则：
- 训练时会合并 `train_dir + val_dir`
- 用 `StratifiedKFold`
- 同时考虑测量时长分层
- 固定 `seed = 42`

稳定观察结论：
- `30s` 一直最难
- `60s` 次难
- `120s / 150s` 明显更稳定


## 五、当前模型与训练配置

当前配置文件中的关键设置：

- `window_feature_dim = 92`
- `use_transformer_branch = false`
- `loss_type = focal`
- `focal_gamma = 2.0`
- `focal_alpha = [1.0, 0.5, 2.0]`
- `meta_features = proba+uncertainty`
- `meta_learner = logreg`

本地常用：
- `num_workers = 0`

5090 常用：
- `batch_size = 512`
- `num_workers = 16`

编译与 AMP：
- `torch.compile(..., backend="aot_eager")`
- `train_ensemble.py` 已加入有限重试与 finite 检查，主要用于防 GTX 1650 本地异常


## 六、当前特征工程状态

### 6.1 已经有的特征

当前已经落地的特征包括：

- SG 导数相关输入
- 对数比值特征
- 小波能量特征
- PCA 得分
- 峰局部对比特征
- 按测量时长加权
- 基于人工审查结果的样本级降权

### 6.2 当前维度

当前总特征维度：
- 基础工程特征：`77`
- PCA：`15`
- 合计：`92`

对应关系：
- `configs/config.json -> model.window_feature_dim = 92`

说明：
- `src/dataset.py` 里的 `_precompute_statistics()` 会自动校验维度
- 如果以后再加特征，必须同步更新 `window_feature_dim`

### 6.3 最近新增的一批物理特征

在提交 `352a3d8 feat: add physics-guided silt features` 中，新加了 5 个特征：

- 净峰面积比值 `net_k / net_th`
- 净峰面积比值 `net_th / net_u`
- 净峰面积比值 `net_k / net_u`
- `compton_curvature`
- `net_peak_fraction`

这一版实验结论：
- 对整体结构有少量帮助
- 但没有实质改善粉土主瓶颈
- 当前最佳结果仍不是这版


## 七、OOF 缓存与 Phase2-only 能力

当前项目已经支持：

- `--oof-path`
- `--phase2-only`
- `--save-oof`
- `--meta-features`

对应文件：
- [`experiments/artifacts/oof_cache.npz`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/oof_cache.npz)

这套能力已经稳定，作用是：
- 完整跑一次 Phase 1 后保存 OOF
- 后续只改 stacking 元特征时，不需要重复跑完整训练

当前已做的防护：
- `allow_pickle=False`
- `config_hash` 校验
- metadata 校验
- OOF finite 检查


## 八、训练后产物与“老三样”

每次完整训练结束后，产物目录在：
- [`experiments/artifacts/`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts)

典型产物：
- `stacking_oof_metrics.json`
- `stacking_oof_report.txt`
- `stacking_oof_predictions.csv`
- `stacking_oof_misclassified.csv`
- `stacking_oof_confusion_counts.png`
- `stacking_oof_confusion_norm.png`
- `stacking_oof_per_class_f1.png`

当前这条项目线约定的“老三样”是：
1. `stacking_oof_metrics.json`
2. `stacking_oof_predictions.csv` 的混淆矩阵
3. `measure_time` 分组准确率


## 九、人工审查 / 软处理这条线目前做到哪了

主文件：
- [`experiments/artifacts/noisy_silt_candidates.csv`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/noisy_silt_candidates.csv)

已做过的轮次：
- [`experiments/artifacts/noisy_silt_candidates_round2.csv`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/noisy_silt_candidates_round2.csv)
- [`experiments/artifacts/noisy_silt_candidates_round3.csv`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/noisy_silt_candidates_round3.csv)

当前口径：
- 这些 `sample_weight` 不是“属于粉土的概率”
- 而是“这个带着粉土标签的样本，值不值得被当成标准粉土强监督学习”

经验结论：
- `round2` 带来了主要收益
- `round3` 仍有一点帮助，但已经明显进入收益递减
- 大规模继续做 `round4 / round5` 不划算


## 十、当前最佳结果与最近几轮结果

### 10.1 当前强基准

目前最值得当作强基准对比的结果是：
- 目录：
  - [`experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260312_224059`](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260312_224059)
- 指标：
  - `Accuracy = 0.7762`
  - `Macro-F1 = 0.7470`

这是当前主要 benchmark。

### 10.2 最近几轮结论

手工审查后 `round3` 方向：
- 结果大约在 `Acc 0.7723`
- `Macro-F1 0.7435`
- 对 persistent hard silt 有一点作用，但没刷新最佳

第一批物理特征：
- 结果大约在 `Acc 0.7728`
- `Macro-F1 0.7418`
- 砂土更强了一点
- 粉土反而略退

结论：
- 第一批物理特征不是完全没信息
- 但没有击中当前主瓶颈


## 十一、误分类归因总结

### 11.1 persistent 粉土错分现在是什么样

现在剩下的顽固粉土错分，已经不是大面积散乱错误，而是集中到少数固定模式：

- 大多集中在 `30s / 60s`
- 高置信错分
- 低熵
- 错分方向稳定：
  - `粉土 -> 粘土`
  - `粉土 -> 砂土`

这说明：
- 它们不是随机波动
- 不是简单训练不够
- 很多样本本身就是边界样本、混合样本或非典型粉土

### 11.2 人工审查真正解决了什么

人工审查这条线主要解决的是：
- 中低置信
- 中等 margin
- 边界模糊但可修复
的那部分粉土错分

而对最顽固的那批 high-confidence hard samples，作用有限。

所以现在的瓶颈已经变成：

**不是继续清洗更多样本，而是要让特征表达更能区分“短时长 + 边界型粉土”。**


## 十二、接下来最推荐做什么

### 12.1 不建议继续做什么

当前不建议优先做：

- 再来一轮大规模 `round4` 样本人工审查
- 再堆一批类似的全局窗口比值特征
- 直接做大规模模型架构重写
- 在没验证局部特征之前，直接切层级策略

### 12.2 当前推荐方向

当前最合理的下一步是：

从“全局窗口比值”转向更贴近顽固粉土错分的特征：

- 净峰-背景分离
- 局部峰形
- 短时长噪声鲁棒

### 12.3 最值得先做的 3 个局部 / 鲁棒特征

后续优先考虑这 3 个：

1. `net_peak_snr`
- 关键思想：`(峰高 - 背景) / 局部噪声`
- 目的：判断短时长条件下主峰到底靠不靠谱

2. `net_peak_area_fraction`
- 关键思想：净峰面积占窗口或全谱的比例
- 目的：把“真实峰强”和“背景抬高”分开

3. `peak_sharpness`
- 关键思想：看峰顶局部是尖还是钝
- 目的：抓住边界样本中“整体相似、局部峰形不同”的区别


## 十三、为什么当前要从全局物理比值转向局部/鲁棒特征

原理上讲：

- 全局窗口比值更擅长区分“大类差异”
- 但当前剩下的错误不是大类没分开，而是边界样本细节没分开

现在这些 persistent 粉土错分的本质更像：
- 峰和背景混在一起
- 局部峰形不稳定
- 短时长噪声把粉土特征淹掉了

所以真正需要回答的问题已经不是：
- “整体上更像哪类”

而是：
- “关键峰到底清不清楚”
- “峰和背景能不能分开”
- “短时长条件下这个峰是否仍可置信”


## 十四、AutoDL 5090 训练操作要点

项目目录：
- `/root/autodl-tmp/gamma/gamma-spectrum-classification`

典型更新流程：

```bash
cd /root/autodl-tmp/gamma/gamma-spectrum-classification
git restore configs/config.json
git restore src/train_ensemble.py
git pull
git log -1 --oneline
```

如果 `git pull` 因网络失败：
- 直接重试

如果 `git pull` 因本地修改冲突失败：
- 通常先 `git restore configs/config.json`
- 需要时再 `git restore src/train_ensemble.py`
- 然后再 `git pull`

服务器训练前常用配置改写：

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


## 十五、当前接手时最短摘要

如果新的开发者只看一段话，应该看这一段：

- 当前主线已经稳定
- 当前最佳仍是人工审查增强后的老版本，不是最新那版物理比值特征
- 大规模继续人工审查已经不划算
- 第一批新物理特征没有击中粉土主瓶颈
- 接下来最值得做的是：
  - 净峰-背景分离
  - 局部峰形
  - 短时长噪声鲁棒

这就是当前项目的真实状态。
