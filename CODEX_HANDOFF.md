# Gamma Energy Spectrum Classification Handoff

> 文档类型：长期交接 / 清空上下文后恢复用  
> 项目目录：`E:\py_project\Gamma Energy Spectrum Label Classification Prediction`  
> 最近重写时间：2026-03-19  
> 当前阶段：方案 A 已基本吃透，正在推进方案 B1；B2/B3 作为后续发展方向已明确  

## 1. 项目总览

这是一个基于 820 通道伽马能谱 `.txt` 文件进行土壤三分类的项目，类别为：

- `0 = 粘土`
- `1 = 砂土`
- `2 = 粉土`

项目核心不是“把一个模型训起来”，而是围绕真实科研目标做持续优化：

- 提升 hardest class `粉土` 的识别能力
- 尽量不明显牺牲总 `Acc`
- 在完整训练和 `Phase2-only` 快速实验之间形成闭环

主入口文件：

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\train_ensemble.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py)

主流程由以下模块构成：

- TriBranch CNN 集成
- GradientBoosting
- XGBoost
- Phase 2 stacking / hierarchical stacking
- OOF cache
- artifacts 导出

## 2. 数据、任务与评估口径

### 2.1 输入与任务

- 每个样本是一个 `.txt`
- 每个样本有 `820` 个能谱通道
- 每个样本有测量时长：`30s / 60s / 120s / 150s`
- 目标是三分类：`粘土 / 砂土 / 粉土`

### 2.2 路径

本机：

- 训练：`E:/data/xunlian`
- 验证：`E:/data/yanzheng`

AutoDL / 5090：

- 训练：`/root/autodl-tmp/gamma/gamma_data/xunlian`
- 验证：`/root/autodl-tmp/gamma/gamma_data/yanzheng`

### 2.3 评估口径

完整训练时会将 `train_dir + val_dir` 合并后做 `StratifiedKFold` 的 OOF 评估。

当前稳定认识：

- `30s` 最难
- `60s` 次难
- `120s / 150s` 较稳

所以当前真正的瓶颈是：

- 短时长样本
- 粉土边界样本

### 2.4 老三样

每次完整训练结束后的固定观察口径是：

1. `stacking_oof_metrics.json`
2. confusion matrix
3. `measure_time` 分组准确率

当前也可以优先使用 artifacts 自动导出的这两个更友好的文件：

- `stacking_oof_metrics_ascii.json`
- `stacking_oof_accuracy_by_measure_time.json`

## 3. 当前关键文件

### 3.1 核心训练文件

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\train_ensemble.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train_ensemble.py)
  - 主入口
  - Phase 1 / Phase 2
  - flat / hierarchical stacking
  - 方案 B1 的 ML 双路径逻辑

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\configs\config.json](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/configs/config.json)
  - 当前主配置
  - 方案 A / B1 开关所在处

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\dataset.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/dataset.py)
  - 数据读取
  - 特征构造
  - DataLoader kwargs

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\train.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/train.py)
  - CNN 单模型训练

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\evaluate.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/evaluate.py)
  - 验证与推理

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\src\artifacts.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/src/artifacts.py)
  - 指标与图表导出

### 3.2 结果与辅助脚本

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\scripts\summarize_artifacts.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/scripts/summarize_artifacts.py)
  - 汇总多个 artifact 的关键结果

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\artifacts](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts)
  - 完整训练与 `Phase2-only` 的所有产物

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\logs\train_ensemble_v2.log](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/logs/train_ensemble_v2.log)
  - 当前本机总日志

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\plot_noisy_silt.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/plot_noisy_silt.py)
  - hard silt 可视化辅助脚本
  - 当前有用户本地未提交修改，不能误带入

## 4. 当前配置与事实

### 4.1 特征维度

- `window_feature_dim = 87` 当前是正常值
- 当前应理解为：
  - 基础工程特征 `72`
  - PCA `15`
  - 合计 `87`

### 4.2 方案 A 当前主配置

方案 A 的定义：

- Phase 1 保持原三分类基模型训练
- Phase 2 改成 hierarchical stacking
- Stage 1：`粉土 vs 非粉土`
- Stage 2：`粘土 vs 砂土`

当前 tuned threshold 关键配置：

- `strategy = hierarchical`
- `meta_features = proba+uncertainty`
- `meta_learner = logreg`
- `stage1_decision = threshold`
- `silt_threshold = 0.43`

### 4.3 B1 开关

新增配置：

- `training.ml_hierarchical_training`

语义：

- `false`：完全沿用方案 A
- `true`：只让 ML 分支进入 B1 两层训练

注意：

- 这个开关必须放在 `training`
- 不能放在 `stacking`
- 原因：`_config_hash()` 会排除 `stacking`，否则 OOF cache hash 感知不到变化

## 5. 已经验证出的结论

### 5.1 flat baseline 的定位

flat baseline 已经通过多轮验证测出稳态：

- `Acc ≈ 0.772`
- `Macro-F1 ≈ 0.742`
- 粉土 `F1 ≈ 0.596`

当前定位：

- `Acc` 导向的对照主线
- 最传统、最好解释的 baseline

历史高点完整训练：

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\artifacts\phase2_phase2_full_proba_uncertainty_20260312_224059](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260312_224059)
- `Acc = 0.7762`
- `Macro-F1 = 0.7470`

但这个更像高点，不像稳态均值。

### 5.2 方案 A 已基本吃透

方案 A 的核心认识已经比较清楚：

- hierarchical 思路是有效的
- 最关键杠杆确实是 Stage 1 的粉土判决
- tuned threshold 可以稳定改善粉土 `F1 / Macro-F1`
- 但很难把完整训练 `Acc` 大幅拉高到新平台

所以方案 A 当前定位是：

- `粉土 F1 / Macro-F1` 导向主线

### 5.3 threshold sweep 结论

已完成两轮 sweep。

粗扫：

- `0.45 / 0.50 / 0.55 / 0.60`

细扫：

- `0.41 / 0.42 / 0.43 / 0.44`

在 `Phase2-only + OOF cache` 的严格可比口径下，当前最优点是：

- `silt_threshold = 0.43`

对应目录：

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\artifacts\phase2_phase2_only_proba_uncertainty_20260318_195149](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_only_proba_uncertainty_20260318_195149)

结果：

- `Acc = 0.7795`
- `Macro-F1 = 0.7536`
- 粉土 `F1 = 0.6247`

为什么选 `0.43`：

- 比 `0.41 / 0.42` 更少把粘土吸进粉土
- 比 `0.44` 又保住了更多粉土
- 是当前最像平衡点的阈值

### 5.4 完整训练下的方案 A

`0.43` 的完整训练代表结果：

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\artifacts\phase2_phase2_full_proba_uncertainty_20260318_225014](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260318_225014)
  - `Acc = 0.7738`
  - `Macro-F1 = 0.7466`
  - 粉土 `F1 = 0.6123`

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\artifacts\phase2_phase2_full_proba_uncertainty_20260319_012523](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/artifacts/phase2_phase2_full_proba_uncertainty_20260319_012523)
  - `Acc = 0.7714`
  - `Macro-F1 = 0.7432`
  - 粉土 `F1 = 0.6043`

均值大约：

- `Acc ≈ 0.7726`
- `Macro-F1 ≈ 0.7449`
- 粉土 `F1 ≈ 0.6083`

结论：

- 对粉土和 Macro-F1 是正向的
- 比默认 `0.50` 更稳
- 是当前最合理的粉土/Macro-F1 主线

### 5.5 项目发展时间线 + 关键 Commit 对照表

#### 5.5.1 flat baseline 建立期
项目最早先把 820 通道伽马能谱三分类的完整训练链路跑通，核心目标是建立一个可复现、可对照的 flat baseline。这个阶段重点是 TriBranch CNN + GB + XGB、OOF cache、Phase 2 stacking 和 artifact 导出，先把“能稳定训练、能稳定评估”做实。

- 关键 commit：`f807ae7`
- 含义：恢复当时最佳 round2 baseline，最接近历史上 `Acc≈0.7762` 的 flat 代码线
- 阶段定位：项目基线，不追求复杂分层，主要用于后续所有方案的对照

#### 5.5.2 瓶颈识别期
随着 flat baseline 稳定后，项目逐渐进入高平台期：总 `Acc` 继续提升变难，但粉土这一 hardest class 一直拖累 `Macro-F1`。这时研究重点开始从“继续堆总分”转向“针对粉土边界问题做结构性改进”。

- 这一阶段没有单一 commit 能完整代表，但它是后续 hierarchical 思路的直接前提
- 核心认知：真正的瓶颈不是训练不通，而是粉土边界样本难分

#### 5.5.3 方案 A 引入期：hierarchical stacking
项目首次把 hierarchical 思想正式引入最终判决层，核心是把三分类拆成两层：先判“粉土 vs 非粉土”，再在非粉土内部判“粘土 vs 砂土”。

- 关键 commit：`78a1e6d`
- 含义：首次加入 hierarchical stacking strategy
- 阶段定位：方案 A 起点，还是早期版，主要回答“分层判决是否有价值”

#### 5.5.4 方案 A 成熟期：Stage 1 threshold
后续项目意识到，hierarchical 是否有效，关键不只是“有没有分层”，而是 Stage 1 到底放多少样本进粉土。因此引入阈值控制，开始系统 sweep `silt_threshold`。

- 关键 commit：`cb484a4`
- 含义：加入 Stage 1 threshold 决策
- 关键结论：默认 `0.50` 偏保守；`0.43` 左右是更稳的平衡点
- 阶段定位：方案 A 进入可精修状态

#### 5.5.5 方案 A 定稿期
经过多轮 `Phase2-only + OOF cache` sweep，以及完整训练复验，方案 A 的角色被明确：它不是为了极限 `Acc`，而是更适合粉土 `F1 / Macro-F1` 的主线。当前最稳定的配置是 `hierarchical + threshold + 0.43`。

- 关键 commit：`08f16cc`
- 含义：刷新 handoff，正式沉淀 hierarchical tuning 结论
- 当前定位：
  - `flat baseline` 作为 Acc 导向对照主线
  - `hierarchical + 0.43` 作为粉土 F1 / Macro-F1 导向主线

#### 5.5.6 B1 验证期：ML 分支层级训练
在方案 A 基本吃透后，项目尝试把 hierarchical 思想前推到 ML 分支训练，形成 B1。目标是让 GB / XGB 也按层级任务学表示，而不仅仅在最终判决层做分层。

- 关键 commit：`7495580`
- 含义：加入 B1 hierarchical ML branch prototype
- 后续关键 commit：`f1f2b5b`
- 含义：增强 B1，加入 B1.1 / B1.2 / B1.3
- B1 的尝试内容：
  - Stage1 / Stage2 独立重采样
  - staged uncertainty / staged meta features
  - class-balanced weighting

#### 5.5.7 B1 结论期：验证未超过方案 A
B1 初版和增强版都没有稳定超过方案 A，说明只改 ML 分支还不够，hardest class 的核心瓶颈可能更偏向 CNN 表征层，而不是 GB/XGB 后端。

- 当前结论：
  - B1 不是主线替代方案
  - 方案 A 继续保留为当前最成熟主线
  - 如果继续冲新上限，更可能考虑 B2，而不是继续细抠 B1

#### 5.5.8 工程化增强期
为了让研究迭代更稳，项目同步补强了工程链路，包括配置拆分、结果摘要、样本降权跨平台修复、artifact 输出可读性和训练性能优化。

- 关键 commit：
  - `71cee74`：训练与数据传递提速
  - `dd2ca40`：修复 artifact 绘图字体 warning
  - `24bfbd4`：增加 console-friendly artifact summaries
  - `c5f067a` / `92f13ec`：补充并增强结果汇总脚本
  - `f3bf2fa`：修复样本级降权跨平台路径匹配
  - `6dcd77c`：统一 noisy_silt 权重优先级，回到 round2 口径

#### 5.5.9 当前总结
- `f807ae7` 是最接近历史最佳 flat baseline 的代码锚点
- `78a1e6d -> cb484a4 -> 0.43 threshold` 是方案 A 的发展主线
- `7495580 -> f1f2b5b` 是 B1 从原型到增强版的完整尝试
- 当前项目最稳妥的主线仍是：
  - `flat baseline` 用于 Acc 对照
  - `hierarchical + 0.43` 用于粉土 F1 / Macro-F1 对照

## 6. 方案 A、B1、B2、B3 的整体路线

### 6.1 方案 A

定义：

- 不改 Phase 1 三分类训练目标
- 只在 Phase 2 引入 hierarchical 决策

作用：

- 已验证有效
- 已基本吃透
- 当前作为粉土/Macro-F1 主线

### 6.2 方案 B1

定义：

- CNN 继续保持三分类训练
- 只把 ML 分支改成两层训练

具体做法：

- Stage 1：`粉土 vs 非粉土`
- Stage 2：在非粉土中训练 `粘土 vs 砂土`
- 最后再合成为三分类概率
- 继续接入现有 Phase 2 stacking

目标：

- 验证“把 hierarchical 思想前推到 ML 基模型训练阶段”是否能带来额外收益

兼容性要求：

- 默认 `ml_hierarchical_training = false` 时完全不破坏方案 A
- 输出仍然是 `(N, 3)` 概率
- OOF cache 和 Phase 2 接口不重写

### 6.3 方案 B2

定义：

- 在 B1 被证明有效的前提下
- 将 hierarchical 训练进一步扩展到 CNN 分支

可能路径：

- 训练一套 `粉土 vs 非粉土` 的 CNN
- 再训练一套 `粘土 vs 砂土` 的 CNN
- 或将当前 CNN 框架升级为支持层级标签目标的输出结构

目标：

- 让最强的表征模型也真正为 hierarchical 目标服务
- 不再只让 ML 分支提前学习 hierarchical

当前状态：

- 还未实现
- 仅作为后续方向
- 是否进入 B2，取决于 B1 的实验结果

### 6.4 方案 B3

定义：

- 全链路层级集成版本
- CNN、GB、XGB、OOF、Phase 2 全部围绕 hierarchical 任务组织

包括但不限于：

- 两层标签体系贯穿整个 Phase 1
- OOF cache 结构支持层级输出
- Phase 2 的 meta 特征与 artifacts 全面对齐层级训练结果

目标：

- 真正把 hierarchical 从“决策层技巧”升级成“全系统训练范式”

当前状态：

- 只作为中长期发展方向
- 不应在 B1 还没结论时提前重构

### 6.5 为什么从 A 走向 B

不是因为方案 A 失败，而是因为：

- 方案 A 已证明 hierarchical 思想本身有效
- 也暴露了它的上限主要卡在“只有决策层在 hierarchical”
- 如果还想冲更高上限，就必须把这个思想前推到基模型训练阶段

## 7. 已完成的代码改动

### 7.1 训练与数据传递提速

提交：

- `71cee74 perf: improve dataloader and device transfer efficiency`

内容：

- DataLoader helper
- `persistent_workers`
- `prefetch_factor`
- `pin_memory`
- `non_blocking` device transfer
- `optimizer.zero_grad(set_to_none=True)`
- `torch.inference_mode()`

### 7.2 字体 warning 修复

提交：

- `dd2ca40 fix: suppress plot glyph warnings in artifacts`

内容：

- 无中文字体时自动退回英文标签
- 解决 `Glyph missing from font(s) DejaVu Sans`

### 7.3 控制台友好 artifacts

提交：

- `24bfbd4 fix: add console-friendly artifact summaries`

内容：

- `stacking_oof_metrics.json` 使用 `utf-8-sig`
- 新增 `stacking_oof_metrics_ascii.json`
- 新增 `stacking_oof_accuracy_by_measure_time.json/csv`

### 7.4 结果汇总脚本

提交：

- `c5f067a feat: add artifact summary script`
- `92f13ec feat: make artifact summaries comparison-friendly`

脚本：

- [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\scripts\summarize_artifacts.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/scripts/summarize_artifacts.py)

### 7.5 B1 原型

提交：

- `7495580 feat: add B1 hierarchical ML branch prototype`

当前核心点：

- 在 `src/train_ensemble.py` 中增加 ML 双路径
- 默认不启用 B1
- 启用后只改变 GB/XGB 的训练方式

## 8. 子代理清单与职责

### 8.1 Hume

- 类型：explorer
- 职责：
  - 梳理 `src/train_ensemble.py` 中 ML / OOF / Phase 2 数据流
  - 辅助定位 B1 最小改动面

### 8.2 McClintock

- 类型：worker
- 职责：
  - 创建 `scripts/summarize_artifacts.py` 初版
  - 负责结果汇总脚本的起始搭建

### 8.3 Parfit

- 类型：worker
- 职责：
  - 完善 `scripts/summarize_artifacts.py`
  - 增强 confusion 和 by-time 的对照输出能力

### 8.4 Meitner

- 类型：explorer
- 职责：
  - 审查 B1 兼容性
  - 识别 residual risks
  - 它不是训练代理

### 8.5 Beauvoir

- 类型：worker
- 职责：
  - 执行本机 B1 第一轮完整训练
  - 当前正在运行的训练任务就是它负责的

## 9. 当前正在进行的事情

当前本机正在跑 B1 第一轮完整训练。

关键信息：

- 训练进程 PID：`33312`
- Python 路径：`C:\Users\14254\.conda\envs\xsypytorch\python.exe`
- 启动时间：`2026-03-19 13:49:36`
- 总日志：
  - [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\experiments\logs\train_ensemble_v2.log](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/experiments/logs/train_ensemble_v2.log)

日志中已确认：

- `ML 层级训练: 启用(B1)`
- `Fold 1/5`
- CNN 已进入训练并出现 Early Stop 记录

注意：

- 日志里有一段 `13:45` 的 smoke start 痕迹
- 当前真实持续运行的是 `13:49:45` 开始的这轮

## 10. 当前工作区状态与注意事项

截至 2026-03-19 当前核对到：

- `CODEX_HANDOFF.md` 已修改但未提交
- `configs/config.json` 已修改但未提交
  - 原因：当前正在为 B1 训练临时设置 `ml_hierarchical_training = true`
- `plot_noisy_silt.py` 有用户本地未提交修改
- 用户已有未跟踪/异常文件：
  - `Path`
  - `able transformer branch via config flag…`
  - `s_sample_multipliers to upweight silt sampling…`
  - `scripts/disable_transformer_branch_exp1.py`

规则：

- 不要擅自回退 `plot_noisy_silt.py`
- 不要清理这些异常未跟踪文件
- 只有用户明确要求时才处理

## 11. 当前 Git 状态

当前本地 `main` 比 `origin/main` 超前 3 个提交：

- `c5f067a feat: add artifact summary script`
- `92f13ec feat: make artifact summaries comparison-friendly`
- `7495580 feat: add B1 hierarchical ML branch prototype`

`origin/main` 当前停在：

- `24bfbd4 fix: add console-friendly artifact summaries`

如果要同步 B1 和汇总脚本，需要：

```powershell
git push origin main
```

但推送前必须确认：

- 不要误带 `plot_noisy_silt.py`
- 不要误动那些用户异常未跟踪文件
- `configs/config.json` 是否处于想提交的目标状态

## 12. 后续建议

### 12.1 如果当前 B1 训练完成

第一步：

- 找最新完整训练 artifact
- 用 [E:\py_project\Gamma Energy Spectrum Label Classification Prediction\scripts\summarize_artifacts.py](/E:/py_project/Gamma%20Energy%20Spectrum%20Label%20Classification%20Prediction/scripts/summarize_artifacts.py) 汇总
- 与方案 A 主线对比

重点看：

- `Acc`
- `Macro-F1`
- 粉土 `precision / recall / f1`
- confusion 摘要
- `measure_time` accuracy

### 12.2 如果 B1 没明显收益

- 先不要立刻进入 B2/B3
- 保留方案 A 作为当前稳定主线
- 将 B1 记为“前推 ML 训练的第一轮验证”

### 12.3 如果 B1 有收益

再考虑：

- 是否进入 B2
- 是否让 CNN 也开始学习层级目标

## 13. 常用命令

查看当前训练进程：

```powershell
Get-Process -Id 33312
```

看总日志尾部：

```powershell
Get-Content experiments\logs\train_ensemble_v2.log -Tail 80
```

汇总某个 artifact：

```powershell
python scripts\summarize_artifacts.py experiments\artifacts\<DIR> --show-confusion --show-time-accuracy
```

列最新 artifacts：

```powershell
dir experiments\artifacts
```

## 14. 一句话恢复上下文

如果上下文被清空，只要先读这份文档，就应立刻知道：

- 方案 A 已基本吃透
- 当前 tuned threshold 是 `0.43`
- B1 已经落地代码但默认不启用
- 当前本机正在跑第一轮 B1 完整训练
- B2/B3 只是后续方向，还未进入实现
- 工作区里有用户自己的未提交/未跟踪文件，不能乱动
