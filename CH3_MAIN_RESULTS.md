# 第三章主结果：全模型对比表

**协议**: FD002/FD004 = V1 统一协议（分工况归一化）；MetroPT-3 = 首月协议  
**评估**: 一次性 test（validation-best checkpoint）；FD 为 raw cycle；MetroPT 为 raw unit overall  
**代码**: `/home/new_work/ch3_mtk_dg_from_scratch`（commit `fff8a4d`）；数据: seed 2026（除注明多 seed 外）

---

## FD002（V1 统一协议，Test，raw cycle）

| 排名 | 模型           | 参数量        | best_ep | MAE       | RMSE      | NASA      |
| -- | ------------ | ---------- | ------- | --------- | --------- | --------- |
| 1  | **MTK-DG**   | **40,940** | 15      | **16.43** | **24.88** | **4,600** |
| 2  | GRU          | 56,449     | 76      | 20.24     | 30.96     | 23,413    |
| 3  | TCN          | 56,449     | 34      | 20.75     | 30.60     | 23,278    |
| 4  | PatchTST     | 115,585    | 35      | 21.78     | 33.81     | 80,942    |
| 5  | iTransformer | 85,697     | 97      | 26.47     | 36.62     | 58,320    |
| 6  | TimesNet     | 25,570     | 137     | 25.06     | 37.07     | 126,285   |

MTK-DG 对最优 baseline（RMSE 口径 TCN）：ΔMAE = −4.32（噪声 ±0.54，8.0×）、ΔRMSE = −5.72（噪声 ±0.42，13.6×）。

## FD004（V1 统一协议，Test，raw cycle）

| 排名 | 模型           | 参数量        | best_ep | MAE       | RMSE      | NASA      |
| -- | ------------ | ---------- | ------- | --------- | --------- | --------- |
| 1  | **MTK-DG**   | **40,940** | 11      | **18.81** | **27.13** | **5,311** |
| 2  | GRU          | 56,449     | 79      | 19.26     | 29.12     | 15,310    |
| 3  | PatchTST     | 115,585    | 19      | 20.64     | 31.55     | 96,634    |
| 4  | TCN          | 56,449     | 18      | 22.27     | 30.25     | 8,082     |
| 5  | TimesNet     | 25,570     | 52      | 25.25     | 34.62     | 28,316    |
| 6  | iTransformer | 85,697     | 36      | 28.56     | 36.48     | 21,604    |

排名按 RMSE。FD004 的 MAE 前两名差距 0.45 在 seed 噪声（±0.74）内：**MTK-DG 与 GRU 统计并列**，待多 seed 裁决。

## MetroPT-3（首月协议，Test，raw unit，overall 指标，seed 2026）

| 排名 | 模型           | 参数量        | best_ep | overall_MAE | overall_RMSE |
| -- | ------------ | ---------- | ------- | ----------- | ------------ |
| 1  | TCN          | 62,145     | 72      | **0.5903**  | 1.9621       |
| 2  | PatchTST     | 204,481    | 83      | 0.6530      | 1.8552       |
| 3  | **MTK-DG**   | **41,892** | 84      | 0.6123      | **1.6266**   |
| 4  | iTransformer | 91,393     | 53      | 0.8415      | 2.0486       |
| 5  | GRU          | 62,145     | 65      | 0.7896      | 2.5494       |
| 6  | TimesNet     | 31,266     | 47      | 1.2994      | 3.6931       |

排名按预注册主指标 **overall_RMSE**（按此口径 MTK-DG 第 1）；TCN 的 MAE 优势 0.022 在三 seed 下被判定为噪声（见下表）。

## MetroPT-3 多 seed 验证（MTK-DG vs TCN，三 seed 配对，协议不变）

| 模型         | overall_MAE（mean±std） | overall_RMSE（mean±std） | 逐 seed oRMSE（2026/2027/2028） |
| ---------- | --------------------- | ---------------------- | ---------------------------- |
| **MTK-DG** | 0.6235±0.0098         | **1.690±0.058**        | 1.63 / 1.70 / 1.74           |
| TCN        | 0.6298±0.0348         | 2.101±0.121            | 1.96 / 2.16 / 2.18           |

配对 per-seed 差值 Δ(MTK−TCN)：

- overall_MAE = −0.006±0.025，方向不一致 → **统计平局**（3-seed 均值 MTK-DG 微弱领先）
- overall_RMSE = −0.411±0.066，**3 seeds 全部同向 → 显著优于**（约 6× seed 噪声）

## 口径备注（表格使用约束）

1. FD 表为 **V1（分工况归一化，6 工况 train-engine-only 拟合，min_std=0.01，未见工况回退全局）** 协议；MetroPT 无工况概念，沿用首月协议。两者不可跨数据集合并排名。
2. FD 的 V1 结果目前为 **单 seed 2026**（多 seed 补齐待执行）；MetroPT 表中 MTK-DG 与 TCN 有 3 seed 证据，其余模型为单 seed。
3. PatchTST（H=50 下 7-patch 实现）与 TimesNet（top-1 周期单 block 简化）为本项目统一协议下的实现，结论限定于该范围，不外推为对原方法的判定。
4. NASA 为指数罚分（低估 exp(|d|/13)、过估 exp(d/10)），FD004 上 seed 方差可达 4.5×，只作参考指标并列报告，不单独用于排名。
5. 所有 run 均为：engine-level/首月划分、train-only 归一化、validation-best 选型、恢复后一次性 test；artifact 含 manifest（scaler identity + git SHA）可追溯，位于 `results/{dataset}/{run_id}/`。
