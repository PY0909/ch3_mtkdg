# CH3 → CH4 交接文档（当前进展快照，供第四章模块构思）

**日期**: 2026-10-14　**状态**: 第三章点预测主干完成，V1 协议定稿，多 seed 主表待补
**代码**: `/home/new_work/ch3_mtk_dg_from_scratch`（commit `ee77d18` 及之前）
**第四章并行工作流已存在**: `mtkdg/models/prob_heads.py`（quantile/gaussian/censored 试点）+ `idea-stage/IDEA_REPORT.md`——本文档为其提供第三章侧的全部实证输入。

---

## 1. 冻结的第三章状态（ch4 的起点基线）

### 1.1 数据协议（V1，已升级为主协议）

| 数据集 | 协议 | 归一化 |
|---|---|---|
| FD002/FD004 | engine-level split（split_seed=2026 固定），H=50，test=每 engine 末端窗口，train/val capped@125，test=官方 raw | **V1 per-condition**：6 工况各拟 scaler（train-engine-only，min_std=0.01，min_group=30，未见工况→全局回退），工况编码=settings 保留 2 位小数 factorize |
| MetroPT-3 | 首月协议（2 月前 80% 时间 train / 后 20% val / 3 月起 test），H=168 P=24 stride=60 | 全局 z-score + min_std=0.01（无工况概念，V1 不适用） |

### 1.2 模型与训练配置（冻结，调参预算=零）

MTK-DG：d_model=64，K=4 时间核，scales=(0.25,0.5,1.0)，dynamic graph 1 层+self-loop，
gated fusion；AdamW lr 2e-4 / wd 1e-4 / cosine(T_max=epochs) / batch 128 / grad-clip 10 /
patience 30(FD)·15(MT)；选型 val_RMSE，test 一次性（runner 强制）。

### 1.3 当前数字（统一协议对照，ch4 对比的 point 基准）

**V1 协议（seed 2026，FD raw cycle）**

| | mtkdg | gru | tcn | patchtst | timesnet | itransformer |
|---|---|---|---|---|---|---|
| FD002 MAE | **16.43** | 20.24 | 20.75 | 21.78 | 25.06 | 26.47 |
| FD002 RMSE | **24.88** | 30.96 | 30.60 | 33.81 | 37.07 | 36.62 |
| FD002 NASA | **4600** | 23413 | 23278 | 80942 | 126285 | 58320 |
| FD004 MAE | **18.81** | 19.26 | 22.27 | 20.64 | 25.25 | 28.56 |
| FD004 RMSE | **27.13** | 29.12 | 30.25 | 31.55 | 34.62 | 36.48 |
| FD004 NASA | **5311** | 15310 | 8082 | 96634 | 28316 | 21604 |

**MetroPT-3（协议不变，3 seeds，mtkdg vs tcn 配对）**：oRMSE 1.690±0.058 vs 2.101±0.121
（3 seeds 同向，差 6× 噪声，**显著**）；oMAE 0.624±0.010 vs 0.630±0.035（平局，均值微弱领先）。
其余 4 模型单 seed 大幅落后（oRMSE 1.855–3.693）。

**收敛特性（V1）**：FD 的 best_epoch 11–137（mtkdg 11/15），早停普遍在 41–150；
V0(global) 下为 43–179。工况偏移移除后优化问题显著变容易。

### 1.4 尚未闭合的第三章事项（不阻塞 ch4 构思，但阻塞论文声明）

1. V1 协议下 FD 多 seed（mtkdg/tcn/gru × 2027/2028 = 12 runs）——FD004 的 mtkdg-vs-GRU
   MAE 之争论裁（当前 Δ=−0.45，噪声内）；FD002 大差距（8–13×）升三 seed 证据。
2. V3（MetroPT 退化通道剔除）/ V4（persistence-skip 头）消融未跑（各 ~2 分钟）。
3. V2 raw 单头已跑：**负结果**（见 §3.1），V2b 双头降级。
4. Loop-2 Round 2 外部复审待执行。

---

## 2. ch4 可直接复用的基础设施（勿重造）

| 设施 | 位置 | ch4 用途 |
|---|---|---|
| DatasetBundle + PerConditionScaler | `mtkdg/data/` | 概率头实验直接同一 bundle；scaler identity 入 manifest 可审计 |
| `y_raw_uncapped`（train/val） | SplitArrays 字段 | **删失似然（Tobit）可直接训练**：capped 标签 + raw 值都在 |
| `val_RMSE_rawtarget` 诊断管道 | runner history | ch4 分位数头的选型指标设计参考（注意 §3.1 的失效警示） |
| 冻结 encoder + 任务头接口 | `mtkdg/models/heads.py` | `forward_train` 多头接口已就绪（dual 模式即多任务先例） |
| 一次性 test / artifact / manifest / git SHA | runner + artifacts | 概率 artifact（coverage/CRPS npz）照抄 schema 扩展 |
| 5 个统一协议 baseline | `mtkdg/models/baselines.py` | ch4 概率 baseline（TCN-Gaussian 等）应同法接入同一 runner |
| 预注册分析计划 | `refine-logs/MATRIX_ANALYSIS_PLAN.md` | ch4 须延续：3 seeds 配对、bootstrap 单位=engine/时间块、零调参 |
| 已测 seed 噪声（V0 口径） | tracker | FD MAE ±0.5–0.9，MetroPT oMAE ±0.01–0.035，FD004 NASA 跨 4.5× |

---

## 3. ch4 设计必须吸收的第三章实证事实

### 3.1 负结果：朴素 raw 回归失败（V2a）

FD004 上 raw 单头（直接回归未截断标签）**全面恶化**：MAE +5.61 / RMSE +6.64 / NASA +152%，
且 val 选型指标（raw-target RMSE）与 test 行为背离。
**含义**：ch4 的 censored 头必须用 Tobit 删失似然（prob_heads.py 已按此设计），
不能退化为"直接回归 raw"；选型指标需在 val 上预注册并验证与 test 一致性。

### 3.2 G4（capped 尾部）在 V1 下大幅缓解但未根除

V0→V1：FD004 NASA 236567→5311（−98%），单引擎主导现象消失。但 >125 区域仍无直接
点监督（train/val 标签 capped）；ch4 的分位数上界/删失头是该区域的**唯一原则性覆盖来源**，
且 FD004 的 GRU-并列现实意味着点预测可挖掘的空间已小——不确定性建模是下一层增益。

### 3.3 误差的异质结构（ch4 不确定性头的建模对象）

- **MetroPT 逐通道**：Oil_level 归一化 MAE 11.3σ、Caudal_impulses 7.4σ（近常数不可预测）
  vs Oil_temperature raw RMSE 4.08（慢趋势可预测）→ **逐通道异方差**必要；
  V3 剔除实验（未跑）的结果将决定输入侧 vs 头侧的处理。
- **时间非平稳**：逐月 oRMSE 1.43–1.81 非单调波动 → 区间宽度应随 regime 变化，
  首月训练的校准在多月 test 上会退化（ch4 校准迁移是核心议题）。
- **horizon 递增误差**：MetroPT 逐 horizon MAE 0.45→0.70 单调增 → 区间应随 horizon 变宽。
- **FD 逐引擎重尾**：NASA 对分散中等过估的指数放大（exp(d/10)）→ ch5 风险不对称损失
  的对齐目标；TCN"MAE 平但 RMSE 差 24%"证明点指标掩盖尾部——CRPS/分位覆盖将更有区分度。

### 3.4 度量与协议纪律（直接继承）

- ch4 主指标建议：FD=CRPS + q05–95/q25–75 覆盖 + 区间宽度（NASA 留给 ch5）；
  MetroPT=逐通道 CRPS + coverage（raw unit 报告）。
- 全部结论三 seed 配对；bootstrap 单位 engine / 时间块；零调参；变体指标切换必须预注册。
- NASA 在 FD004 的 seed 方差 4.5×——任何以 NASA 为选型指标的设计先测其方差。

---

## 4. 建议的 ch4 构思输入（连接已存在的 prob_heads 试点）

`prob_heads.py` 三头与本章证据的对齐度检查：

| 试点头 | 第三章证据支持 | 需补的设计点 |
|---|---|---|
| QuantileHead（11 分位+crossing 罚） | ✅ horizon/通道异方差、逐通道不可预测性 | 选型指标（pinball val loss？）预注册；与 V2a 背离教训对照 |
| GaussianHead（NLL） | ✅ 基线形态 | 异方差输入（逐通道 obs_ratio/gate 权重已在 encoder 输出可复用） |
| CensoredHead（Tobit） | ✅✅ 直接对准 G4；V2a 负结果排除了朴素替代 | 右删失阈值=125 固定；val 上无删失子集做覆盖检验 |

冻结 encoder 的表示（z、gate_weights、fusion_alpha、adjacency 已从 forward 暴露）
可直接作为不确定性调制的条件——这是第三章结构留给第四章的最大红利。

---

## 5. 快速复现命令（ch4 实验起点）

```bash
# V1 主协议 point 基线（ch4 对照）
python scripts/run_experiment.py --dataset fd004 --seed 2026 --device cuda \
  --epochs 200 --patience 30 --batch-size 128 --lr-scheduler cosine \
  --norm-mode per_condition --evaluate-test --run-id <rid> ...
# 结果汇总
python scripts/collect_results.py
# 全部 artifact: results/{ds}/{rid}/ （manifest 含 scaler identity + git SHA）
```
