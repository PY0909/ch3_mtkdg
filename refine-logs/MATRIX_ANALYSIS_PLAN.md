# 矩阵统计分析计划（预注册，冻结于多 seed 矩阵启动前）

Round 2 评审冻结门第 1/4/5 项。本文件在看到任何多 seed 结果**之前**写定；
矩阵启动后不得更改（如确需更改，必须在新文件中记录变更理由与时间）。

## 1. 种子与聚合

- 训练 seed 列表：**2026 / 2027 / 2028**（3 seeds）。`split_seed` 固定 2026
  （全部 seed 共享同一 engine 划分与 MetroPT 时间边界；测训练方差，不测划分方差）。
- 主聚合：每指标报告 **mean ± sample std**（ddof=1，n=3）。
- FD004 NASA 等重尾指标额外报告中位数（mean 对单点灾难敏感，须两者并列）。
- 逐 seed 结果全部入表（附 A），不隐藏任何 seed。

## 2. 模型排序与比较

- 排序基于 **paired per-seed differences**（同 seed 下两模型指标差，
  n=3 的配对样本），不用跨 seed 平均值的独立比较。
- 主检验：配对符号一致性（3 seeds 同向）+ 配对 t 检验（n=3，仅作描述性参考，
  不做 0.05 阈值的"显著性"声明——n=3 无统计功效支撑）。
- 论文表述：3 seeds 同向且 mean 差 > 各自 seed-std 时方可写"在此协议下
  稳定优于"；否则写"均值占优但 seed 间不一致"。

## 3. 置信区间

- Bootstrap 95% CI（BCa，10000 次重采样），重采样单位：
  - FD002/FD004：**engine**（窗口高度重叠，不得以窗口为单位）；
  - MetroPT-3：**时间块**（按自然周分块）。
- Bootstrap 在每个 seed 内做，再跨 seed 取中位区间；不跨 seed 混合重采样。

## 4. 失败与发散策略

- run 失败：先修复环境重跑一次；仍失败则该 (model, dataset, seed) 记为缺失，
  报告实际 n，不做插补。
- 发散（train_loss 变 NaN 或 val_RMSE > 2× 初始值）：保留并在附表标注，
  计入 mean；不删除。发散率本身作为稳定性指标报告。

## 5. 早停与训练预算

- 报告每模型 best_epoch 的分布（min/median/max over seeds）。
- 冻结预算：MTK-DG FD 200ep / MetroPT 100ep；baseline FD 150ep / MetroPT 100ep；
  全部 cosine + AdamW(2e-4, wd 1e-4) + batch 128 + grad-clip 10 + patience(30/15)。
- 调参预算：零（所有模型使用冻结超参，未做 per-model 搜索；这一事实写入论文）。

## 6. 主指标（预注册口径）

- FD002/FD004 主任务声明为 **capped-RUL 预测**（train/val 目标 capped@125）；
  raw-RUL test 指标作为**外部诊断**报告，不得表述为对全部剩余寿命的
  无条件估计能力。高 RUL 失败分析（true>125 引擎残差）进正文讨论。
- NASA 与 MAE/RMSE 并列报告并解释口径差异（NASA 指数罚被分散中等过估主导）。
- MetroPT 主指标 overall_RMSE（raw unit）；overall_MAE 并列；逐月分解与
  归一化空间 per-sensor 误差为诊断，不聚合为均衡性声明。

## 7. 消融矩阵（随后启动，A0–A8 按 plan §8.3）

每消融至少 1 seed（2026），显著项补 3 seeds；参数量匹配对照（A8）在
baseline 矩阵之后执行。

## 8. 启动门清单（Round 2 评审）

- [x] 本分析计划冻结
- [x] baseline 预算与参数量已记录（manifests + final_matrix_summary.json）
- [x] 同 seed 重复运行检查（2026-10-14）：metropt3 × {mtkdg, tcn} 两次执行
      逐位一致（|ΔMAE|=|ΔRMSE|=0.0000）——非确定性 CUDA 执行在本栈上未实际发生，
      run-to-run 变差 = 0 << seed 变差，矩阵方差政策可辩护
- [ ] 干净 commit 归档 + 环境记录（pip freeze、GPU/driver）
- [ ] 术语冻结：capped-RUL 主任务、raw 为诊断；矩阵结果前禁止 superiority 声明

## 9. Loop-2 提升消融（预注册于 2026-10-14，先注册后运行）

| 变体 | 内容 | 选型指标（validation-only） | 对照 |
|---|---|---|---|
| V1 per-condition norm | FD 特征按 6 工况分组归一化（train engines 内拟 per-condition scaler，min_std=0.01；未见工况回退全局） | val_RMSE（capped，与主协议一致） | V0=global |
| V2 dual raw-RUL head | 共享 encoder + capped 头 + raw 辅助头，loss=MSE_capped+MSE_raw；test 用 raw 头输出 | val_RMSE_rawtarget（该变体的估计对象即 raw） | V0 |
| V3 metropt exclude-degenerate | train 内 raw std<0.05 的通道从 encoder 输入剔除（mask=0），输出头不变 | val_RMSE（overall，主协议一致） | V0 |
| V4 metropt persistence-skip | StateHead 预测增量：y=x_last+f(z) | val_RMSE | V0 |
| V5 fd004 budget/capacity | epochs 300；d_model 128 | val_RMSE | V0（200ep/d64） |

规则：全部单 seed 2026 起跑；若 V1/V2 显著（Δval 超出 P0 测得的 seed 噪声），补 seed 2027/2028
并在采纳前对全部模型统一重跑；test 每 run 仍一次性。
