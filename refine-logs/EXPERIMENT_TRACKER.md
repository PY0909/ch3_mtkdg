# EXPERIMENT_TRACKER — Chapter 3 MTK-DG 单 seed 实验

Plan 来源: /home/new_work/kaf_profiti/plan/chapter-03-mtk-dg-fcstgnn-plan.md §7.1
共同配置: seed=2026, batch=128, lr=2e-4, wd=1e-4, patience=10, d_model=64,
K=4 kernels, scales=(0.25,0.5,1.0), graph=dynamic(1层), fusion=gated,
checkpoint=val_RMSE, test=一次性（显式 --evaluate-test）
Backend: local, RTX 4060 Ti 8GB, conda env torch23

| Run ID | Dataset | Epochs | Status | val_RMSE(best) | Test 指标 | Notes |
|---|---|---|---|---|---|---|
| fd002_mtkdg_seed2026_20261013_200407 | fd002 | 80 (best=80) | DONE | 20.98 | MAE=22.31, RMSE=30.65, NASA=10953.6 | 末段仍在改善，建议加 epochs |
| fd004_mtkdg_seed2026_20261013_200407 | fd004 | 80 (best=79) | DONE | 20.04 | MAE=27.21, RMSE=37.25, NASA=245004.5 | NASA 对大过估敏感，见 per_engine CSV |
| metropt3_mtkdg_seed2026_20261013_200407 | metropt3 | 50 (best=49) | DONE(r1,保留) | 0.7896 | overall_MAE=0.6199, overall_RMSE=1.6325 | cosine 之前；常数通道未加下限 |
| fd002_mtkdg_seed2026_r2_20261013_211500 | fd002 | 200 cosine (best=133, 早停163) | DONE | 20.81 (capped) | MAE=22.20, RMSE=30.66, NASA=11113.6 | W1 收敛重训；test 与 r1 持平 |
| fd004_mtkdg_seed2026_r2_20261013_211500 | fd004 | 200 cosine (best=179) | DONE | 19.19 (capped) | MAE=26.38, RMSE=36.63, NASA=236566.9 | test 与 r1 持平 |
| metropt3_mtkdg_seed2026_r2_20261013_211500 | metropt3 | 100 cosine (best=84, 早停99) | DONE | 0.7875 | overall_MAE=0.6123, overall_RMSE=1.6266 | 含 min_std=0.01 修复；逐月/归一化分解已产出 |

## 预注册决策（Round 1 评审后，2026-10-13）

### Seed vs split_seed 政策（W5）
- `split_seed` 对所有训练 seed 固定为 2026：三个 seed（2026/2027/2028）共享同一
  engine 划分与同一 MetroPT 时间边界，多 seed 测的是训练方差（初始化/批序），
  不是划分方差。FD bootstrap 单位 = engine，MetroPT bootstrap 单位 = 时间块（plan §8.1）。
- MetroPT 无 engine 划分，`split_seed` 对其无效（时间协议固定）。

### MetroPT 退化通道方差下限（W3，post-first-look 变更，保留旧 run）
- 决策：`TrainOnlyStandardScaler(min_std=0.01)`，常量 `METROPT_MIN_STD=0.01`。
- 原因：`Oil_level` 在 train 内 std=0，scaler 下限 1e-4 会把 test 真实波动放大为
  10^3 sigma 输入进入动态图；MPG/LPS 近退化。raw-unit 指标对此不可见
  （Oil_level 仅占总 MSE ~0.3%），归一化空间 per-sensor 指标已加入
  per_sensor_metrics.csv（MAE_norm/RMSE_norm 列）。
- 影响范围：仅 MetroPT；FD002/FD004 已显式删除常数通道，不启用下限。
- 归一化 identity 已包含 min_std（manifest 可审计）。

### 结果表脚注（W2）
- FD 的 val_RMSE/val_MAE/val_NASA 均对 **capped(125) 目标** 计算；test 指标对
  **官方 raw 目标** 计算。两者不可直接相减解释为泛化 gap。
  history/validation_metrics.json 中的 `val_RMSE_rawtarget` 是对未截断目标的
  监控诊断，不参与 checkpoint 选择（选择仍按协议用 capped val_RMSE）。

### 协议备注（r2 矩阵阶段，2026-10-13 晚）
- **cudnn 确定性标志**：`set_seed` 不再强制 `cudnn.deterministic=True`。实测该标志使
  conv1d 路径慢 ~130x（TCN/TimesNet/PatchTST 等 baseline 卷积受影响；MTK-DG 的
  linear/bmm 不受影响）。全部 RNG 流仍由 seed 固定（数据顺序/初始化完全可复现），
  仅卷积核在跨 GPU 架构时不保证逐位一致。已完成的 r1/r2 MTK-DG run 均在旧标志下
  产出，artifact 有效，不受影响。
- **baseline FD epochs=150**（cosine，patience=30）：五个 baseline 均为标准结构，
  收敛远早于 150；MTK-DG FD 用 200 epochs（best=133/179 时已收敛）。选择规则
  （val_RMSE）、数据协议、normalizer、evaluator、一次性 test 完全一致。
- **TimesNet 简化声明**：保留核心机制（FFT 主周期检测 → 2D 折叠卷积 → 展开摘要），
  使用 top-1 周期、单 block、无 inception 多尺度；非官方完整实现，论文中如实描述。

## 最终矩阵（15/15 完成，seed=2026，统一协议）

Test 指标（FD: raw cycle MAE/RMSE/NASA；MetroPT: raw unit overall_MAE/overall_RMSE）：

| dataset | mtkdg(r2) | tcn | gru | patchtst | timesnet | itransformer |
|---|---|---|---|---|---|---|
| fd002 MAE | **22.20** | 26.18 | 27.88 | 35.03 | 30.34 | 34.57 |
| fd002 RMSE | **30.66** | 35.24 | 36.35 | 44.94 | 40.02 | 43.73 |
| fd002 NASA | **11114** | 51961 | 44057 | 257773 | 68331 | 92903 |
| fd004 MAE | **26.38** | 30.43 | 35.75 | 38.33 | 34.51 | 40.03 |
| fd004 RMSE | **36.63** | 38.66 | 45.07 | 47.86 | 42.92 | 50.21 |
| fd004 NASA | 236567 | **51590** | 218913 | 227320 | 74164 | 1979609 |
| metropt oMAE | 0.612 | **0.590** | 0.790 | 0.653 | 1.299 | 0.842 |
| metropt oRMSE | **1.627** | 1.962 | 2.549 | 1.855 | 3.693 | 2.049 |

诊断要点：
- MTK-DG 单 seed 在 3 个数据集的 MAE/RMSE 均排第 1；FD004 NASA 第 4（TCN 第 1，
  机制见 per-engine 诊断）；MetroPT MAE 第 2（TCN 第 1，差 0.022）。
- baseline 全部在 150/100 epochs 内收敛（best_ep 43–144）。
- 全部为单 seed，禁止写成论文结论；多 seed 与消融未启动。

## V1 统一重跑（per-condition 归一化，6 模型 × FD002/FD004，seed 2026，2026-10-14）

| dataset | mtkdg | tcn | gru | patchtst | timesnet | itransformer |
|---|---|---|---|---|---|---|
| fd002 MAE | **16.43** | 20.75 | 20.24 | 21.78 | 25.06 | 26.47 |
| fd002 RMSE | **24.88** | 30.60 | 30.96 | 33.81 | 37.07 | 36.62 |
| fd002 NASA | **4600** | 23278 | 23413 | 80942 | 126285 | 58320 |
| fd004 MAE | **18.81** | 22.27 | 19.26 | 20.64 | 25.25 | 28.56 |
| fd004 RMSE | **27.13** | 30.25 | 29.12 | 31.55 | 34.62 | 36.48 |
| fd004 NASA | **5311** | 8082 | 15310 | 96634 | 28316 | 21604 |

关键结论：
1. V1 对**所有**模型都是大改进（MTK-DG RMSE −19/−26%，TCN −13/−22%）；V1 升级主协议成立。
2. MTK-DG 在两个数据集的 MAE/RMSE/NASA **全部第 1**；FD002 Δ vs 最优 baseline 超噪声
   （MAE −4.32 vs ±0.54，RMSE −5.73 vs ±0.42）；FD004 MAE 差距（−0.45 vs GRU）在噪声内、
   RMSE（−1.99 vs ±1.20）略超噪声 —— FD004 上 MTK-DG 与 GRU 基本并列。
3. FD004 NASA 结构性修复：V0 236567（单引擎 85%）→ V1 5311，且不再被单引擎主导
   （capped 尾部问题随工况偏移的移除大幅缓解，G4 部分闭合）。
4. 收敛普遍提前（best_ep 11–137 vs V0 的 43–179），早停有效节省算力。
5. 单 seed 2026；FD004 上 MTK-DG vs GRU 需补 seed 定论。

状态：V2b/V3/V4/V5 暂停——在 V1 主协议定稿后重新设计（旧口径下的结果不再可比）。
