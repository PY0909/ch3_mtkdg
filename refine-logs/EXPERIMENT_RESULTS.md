# Initial Experiment Results — Chapter 3 MTK-DG 单 seed

**Date**: 2026-10-13
**Plan**: /home/new_work/kaf_profiti/plan/chapter-03-mtk-dg-fcstgnn-plan.md §7.1
**Code**: /home/new_work/ch3_mtk_dg_from_scratch（commit 5bfc00b，pre-deployment review 无 BLOCKING）
**Backend**: local RTX 4060 Ti 8GB, conda env torch23, seed=2026

## Results by Milestone

### M0: Sanity — PASSED
- 103 项 pytest 全绿；三数据集真实数据 CPU smoke + CUDA 1-epoch 路径验证通过。

### M1: 主方法单 seed（MTK-DG, d_model=64, K=4, dynamic graph, gated fusion）

| Run ID | Dataset | Epochs (best) | val_RMSE | Test MAE | Test RMSE | Test NASA | Params | Train 时长 |
|---|---|---|---|---|---|---|---|---|
| fd002_mtkdg_seed2026_20261013_200407 | FD002 | 80 (best=80) | 20.98 | 22.31 | 30.65 | 10953.6 | 40940 | 351.8s |
| fd004_mtkdg_seed2026_20261013_200407 | FD004 | 80 (best=79) | 20.04 | 27.21 | 37.25 | 245004.5 | 40940 | 408.3s |
| metropt3_mtkdg_seed2026_20261013_200407 | MetroPT-3 | 50 (best=49) | 0.7896* | 0.6199* | 1.6325* | n/a | 41892 | 28.8s |

\* MetroPT 指标为 raw-unit overall（15 传感器 × 24 horizon 全元素平均）；FD 指标为 raw-cycle。

> **脚注（Round1-W2）**：FD 的 val 指标对 capped(125) 目标计算，test 指标对官方 raw 目标计算，
> val→test 差值不能读作泛化 gap；val_RMSE_rawtarget（未截断目标诊断）自 r2 轮起记录。
> 本表为 r1 轮（cosine 之前）的运行，保留作对照；r2 轮见下文。

Artifact 完整性：三个 run 全部含 config/manifest/checkpoint_best/checkpoint_last/
history/training_log.csv/training_curves.png/validation_metrics/test_metrics/
predictions_test.npz/targets_test.npz + FD per_engine_metrics.csv+rul_prediction.png /
MetroPT per_sensor_metrics.csv+state_prediction.png。targets_test.npz 已由独立审查
确认与官方 ground truth 逐值一致；test 均为 val-best checkpoint 的一次性评估。

## Observations（非结论）

1. **三个 run 的 best epoch 都落在训练末端**（80/80、79/80、49/50），validation 曲线
   仍在缓慢下降：当前 epochs 配置未收敛，下一轮可增加 epochs 或加 lr schedule。
2. FD004 的 val NASA 波动大（10^4–10^5 量级跳动），test NASA=245k 明显高于 FD002：
   NASA 是指数罚分，对个别大过估 engine 极敏感，per_engine_metrics.csv 可定位残差
   最大的引擎；这是单 seed 诊断信号，不能外推为模型优劣结论。
3. FD 两个任务的 test Bias 为负（-7.2 / -9.0 cycles，即系统性低估 RUL），
   与 capped-RUL 训练目标的已知行为一致，第四章校准阶段需要关注。
4. MetroPT val(首月末 20%) 与 test(3–8 月) 差距明显（0.79 vs 1.63 raw RMSE）：
   与"首月训练、多月外推"的协议预期一致（时间外推存在分布漂移）。

## 边界与禁止事项遵守

- 未做任何 baseline 对比；以上数字不得表述为"优于"任何模型（无对照 artifact）。
- 单 seed 诊断性结果，正式多 seed（2026/2027/2028）与消融未启动。
- normalizer 全部 train-only；窗口不跨 split；FD002/FD004 分开训练分开报告。

## Summary
- 3/3 must-run 完成，管线与协议全部按预注册执行
- Main result: 单 seed 诊断结果已产出（见上表），无法判定 positive/negative（无对照）
- Ready for /auto-review-loop: 需先补 baseline 与多 seed，当前仅可做单 seed 诊断审阅

## Next Step
1. 延长 epochs（如 150）+ 可选 cosine schedule 重跑单 seed，确认收敛后再冻结超参
2. 多 seed × 3 数据集；随后 baseline（TCN/GRU/PatchTST…）与 A3/A4/A5/A7 消融
3. → /auto-review-loop "chapter 3 point prediction"
