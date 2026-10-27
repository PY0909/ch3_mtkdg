# 第三章主结果：3-seed 同硬件版（2026-10-21）

**协议**: FD002/FD004 = V1 统一协议（分工况归一化，engine-level split，split_seed=2026）；MetroPT-3 = 首月协议  
**评估**: 一次性 test（validation-best checkpoint）；FD 为 raw cycle；MetroPT 为 raw unit overall  
**seeds**: 2026/2027/2028（训练 seed；划分固定），mean±std 为样本标准差；配对 Δ 为逐 seed 相减  
**代码**: <https://github.com/PY0909/ch3_mtkdg> （commit `c933b41` 起含 baseline delta 头）；**全部 run 在同一张 RTX 3090 上产出**  
**明细**: `refine-logs/CH3_3SEED_AGGREGATE.md`（`scripts/aggregate_3seed.py` 自动生成，含每张表的逐 seed 配对与同向性）

> 旧版（单 seed 2026、原机 4060 Ti）的 6 模型表见 §6 附录；本页主表只收录三 seed 同硬件结果。

---

## 1. FD002 / FD004 主表（Test，raw cycle，3 seeds）

> 2026-10-27 补齐批后为**六模型全 3-seed** 表。三批（20261020 / 20261026 / 20261027_extra）均产出自
> **同一 RTX 3090 实例**（AutoDL 关机重开不换宿主，端口 43676 未变）。批间复现校验：fd002 MTK-DG
> seed2026 相隔一周重跑**逐位一致**（16.4234/24.8765/4596.19，含指数放大的 NASA 分数），管线完全确定。

| 数据集   | 模型             | 参数量    | MAE            | RMSE           | NASA             |
| ----- | -------------- | ------ | -------------- | -------------- | ---------------- |
| FD002 | **MTK-DG**     | 40,940 | **16.37±0.15** | **24.75±0.29** | **4,596±162**    |
| FD002 | GRU            | 56,449 | 19.31±0.16     | 29.29±0.45     | 13,451±2,852     |
| FD002 | TCN            | 56,449 | 20.52±0.22     | 30.01±0.50     | 18,845±3,911     |
| FD002 | PatchTST       | 115,585 | 21.93±0.22     | 33.61±0.71     | 63,532±25,246    |
| FD002 | iTransformer   | 85,697 | 26.54±0.44     | 36.55±0.72     | 55,031±11,829    |
| FD002 | TimesNet       | 25,570 | 25.24±0.25     | 37.43±0.45     | 149,912±30,912   |
| FD004 | **MTK-DG**     | 40,940 | **18.42±0.38** | **26.50±0.81** | **4,937±541**    |
| FD004 | GRU            | 56,449 | 19.64±0.19     | 29.47±0.87     | 17,174±6,922     |
| FD004 | PatchTST       | 115,585 | 20.29±0.20     | 30.26±1.04     | 47,116±49,140    |
| FD004 | TCN            | 56,449 | 22.21±0.25     | 30.13±0.17     | 7,763±391        |
| FD004 | TimesNet       | 25,570 | 25.23±0.32     | 34.85±0.34     | 31,630±2,923     |
| FD004 | iTransformer   | 85,697 | 28.29±0.15     | 36.09±0.32     | 21,242±679       |

（表按 RMSE 排序展示；TimesNet 25,570 参数为统一协议简化实现（单周期 block）所致，限定见 §5.4。）

配对差值 Δ(MTK-DG − baseline)，逐 seed 相减（全部 10 组 **3/3 同向**）：

| 数据集   | 对比               | ΔMAE            | ΔRMSE           |
| ----- | ---------------- | --------------- | --------------- |
| FD002 | vs GRU           | −2.94±0.02      | −4.54±0.46      |
| FD002 | vs TCN           | −4.15±0.28      | −5.26±0.45      |
| FD002 | vs PatchTST      | −5.56          | −8.86±0.47      |
| FD002 | vs TimesNet      | −8.87          | −12.68±0.72     |
| FD002 | vs iTransformer  | −10.17         | −11.79±1.00     |
| FD004 | vs GRU           | −1.21±0.29      | −2.97±0.14      |
| FD004 | vs TCN           | −3.78±0.27      | −3.63±0.64      |
| FD004 | vs PatchTST      | −1.87          | −3.76±0.46      |
| FD004 | vs TimesNet      | −6.81          | −8.36±0.82      |
| FD004 | vs iTransformer  | −9.87          | −9.60±0.63      |

（vs PatchTST/TimesNet/iTransformer 的 ΔMAE 为逐 seed 均值，明细见 `refine-logs/CH3_3SEED_AGGREGATE.md` 与各 run `test_metrics.json`。）

## 2. MetroPT-3（Test，raw unit overall，3 seeds）

### 2.1 关键事实：direct 头模型未超过零参数 persistence 基线

| 模型                               | overall_MAE   | overall_RMSE  |
| -------------------------------- | ------------- | ------------- |
| **persistence（ŷ = x_last，0 参数）** | **0.4825**    | 1.6683        |
| MTK-DG (direct)                  | 0.6235±0.0098 | 1.6901±0.0577 |
| TCN (direct)                     | 0.6298±0.0348 | 2.1010±0.1206 |

`scripts/persistence_baseline.py` 以完全相同的窗口/归一化/raw-unit 口径计算。原主表（§6）中所有 direct 头模型在 oMAE 上都不如"复制最后一个观测值"，oRMSE 上 MTK-DG 仅持平。**MetroPT 主表必须含此行，且 direct 头不能作为主协议。**

### 2.2 persistence-skip（delta）头：全模型同头，3 seeds

ŷ = x_last + f(z)，baseline 与 MTK-DG 使用完全相同的 StateHead（`--state-head-mode delta`）。

| 模型           | overall_MAE       | overall_RMSE      | ΔRMSE vs persistence         |
| ------------ | ----------------- | ----------------- | ---------------------------- |
| **MTK-DG**   | **0.4523±0.0185** | **1.2708±0.0284** | **−0.398（唯一优于 persistence）** |
| iTransformer | 0.5904±0.0466     | 1.7329±0.0611     | +0.065                       |
| TCN          | 0.5300±0.0353     | 1.7534±0.1460     | +0.085                       |
| PatchTST     | 0.6161±0.1990     | 1.8316±0.6079     | +0.163                       |
| GRU          | 0.6708±0.0076     | 2.2081±0.0432     | +0.540                       |
| TimesNet     | 0.7076±0.0119     | 2.2709±0.0707     | +0.603                       |

- MTK-DG vs TCN 配对 ΔoRMSE = −0.483±0.17（3 seeds 同向）；vs persistence −24% RMSE / −6% MAE。
- 五个 baseline 即使给了 delta 头仍劣于 persistence：首月训练→后续月份 test 的分布漂移下，它们学到的残差不泛化（逐月指标见各 run 的 `per_month_metrics.csv`）。
- **建议**：MetroPT 主协议改为 delta 头（所有模型统一），direct 头结果降为 §6 附录 + 消融 V4 的对照。此变更须在论文中作为"post-first-look 协议修订"如实声明。

## 2A. MetroPT delta 头补跑批（2026-10-26，**论文口径**）

§3/§4 的 MetroPT 列是 direct 头批次（2026-10-20）产出，与论文"direct 结果不放"口径冲突；2026-10-26 在同一 RTX 3090 实例上以 `--state-head-mode delta` 重跑全部消融变体与 G1–G4（`dl*` 变体名，批内配对，**不与 direct 批混表**）。脚本 `scripts/run_metropt_delta_family.sh`，聚合 `scripts/aggregate_delta_family.py`（commit `6abeca5`），明细 `refine-logs/CH3_DELTA_FAMILY_AGGREGATE.md`。

**批间复现验证**：本批 A0（dla0）= 0.4523±0.0185 / 1.2708±0.0284，与 20261020 批 v4deltaskip **逐位一致**（同一实例相隔六天的两次独立训练）——管线完全确定、无 cuDNN 非确定性问题，故本批消融可与主表 MTK-DG 直接衔接。

### 2A.1 结构消融（delta 头，ΔoRMSE vs dla0，逐 seed 配对）

| 变体                             | oMAE            | oRMSE           | Δ vs dla0                   |
| ------------------------------ | --------------- | --------------- | --------------------------- |
| dla0 全模型（dynamic + gated）     | 0.4523±0.0185   | 1.2708±0.0284   | —                           |
| graph = none                   | 0.4974±0.0198   | 1.3579±0.0134   | **+0.087±0.019（同向，~4.5σ）** |
| fusion = add                   | 0.4627±0.0264   | 1.3095±0.0421   | +0.039±0.014（同向，~2.8σ）     |
| fusion = self_only（≡ no_graph） | 0.5109±0.0098   | 1.3424±0.0202   | +0.072±0.020（同向）           |

解读（delta 口径，相对 direct 批的三点变化）：

1. **图分支结论保持且更稳**：去掉图 +0.087（3 seeds 同向，~4.5σ），幅度小于 direct 批（+0.225）但显著性相同。论文创新一的 MetroPT 证据改用本列。
2. **gated vs add 从"无差别"变为"小幅一致优势"**：direct 批 +0.026（<1σ）、FD 两数据集均在噪声内；delta 批 +0.039（3/3 同向，~2.8σ）。论文措辞：门控融合在 MetroPT 上小幅一致优于直接相加，在 FD 上无差别——不写"门控显著提升"。
3. self_only 与 graphnone 数学上同构（输出仅含 z_self），数值差异来自随机流消耗不同；论文表中只保留 graph = none 一行。

### 2A.2 MC-Graph 门实验（delta 头）：clean + 损坏稳健性

| 配置             | clean RMSE     | Δclean vs dla0         | ch_drop@0.3 | time_block@0.3 |
| -------------- | -------------- | ---------------------- | ----------- | -------------- |
| dla0           | 1.2708±0.0284  | —                      | +55.7%±8.4  | +8.4%±1.7      |
| dlg1 robust    | 1.2390±0.0297  | **−0.032±0.002（同向）**  | +50.6%±2.2  | +4.3%±1.6      |
| dlg2 +consist  | 1.2381±0.0286  | **−0.033±0.002（同向）**  | +50.6%±2.0  | +4.2%±1.6      |
| dlg3 +avail    | 1.2694±0.0152  | −0.001±0.027（不同向）     | +46.6%±0.5  | +4.1%±1.0      |
| dlg4 完整        | 1.2664±0.0156  | −0.004±0.027（不同向）     | +46.8%±0.5  | +4.2%±1.0      |

解读：

1. **delta 头的 channel_dropout 基线大幅变脆（31.3%→55.7%）且 G 配置改善有限（→47%）**：机制清楚——delta 头以最后观测 x_last 为持久性锚，channel_dropout 整通道置零恰好摧毁该锚，预测随锚一起坍缩。这是 persistence-skip 的结构性代价，论文必须如实报告；FD 不受影响（RUL 头无锚）。
2. **time_block 收益保持**：8.4%→4.2%（约减半），与 direct 批方向一致。
3. **robust-train 在 delta 头下 clean 反而同向改善**（dlg1/dlg2 −0.032，direct 批为噪声内）——"零代价"升级为"小幅正收益"；G3/G4 的 availability 编码把 ch_drop 再压 ~4 个点（50.6→46.6），仍属次要组件。
4. 论文稳健性叙述分工：FD 列沿用 §4（20261020 批，time_block 6× 压缩为主证据）；MetroPT 列用本表（time_block 减半 + 锚脆弱性的诚实披露）。

## 3. 结构消融（MTK-DG，3 seeds，ΔRMSE vs A0，逐 seed 配对）

> **口径注**：本节 MetroPT 列为 direct 头批次（2026-10-20）。论文中 MetroPT 消融/稳健性一律采用 §2A 的 delta 口径；本节 MetroPT 列仅作实验历史保留。FD 两列不受影响，仍是论文口径。

| 变体                             | FD002           | FD004           | MetroPT (oRMSE)          |
| ------------------------------ | --------------- | --------------- | ------------------------ |
| A0 全模型（dynamic + gated）        | 24.75±0.29      | 26.50±0.81      | 1.690±0.058              |
| graph = static                 | −0.12±0.38（不同向） | −0.42±0.54（不同向） | +0.037±0.04（不同向）         |
| graph = none                   | +0.53±0.31（同向）  | +0.23±0.53（不同向） | **+0.225±0.08（同向）**      |
| fusion = add                   | −0.03±0.30（不同向） | +0.08±0.39（不同向） | +0.026±0.02（同向，<1σ）      |
| fusion = self_only（≡ no_graph） | +0.51±0.47（同向）  | +0.05±0.67（不同向） | **+0.137±0.16（同向）**      |
| V3 剔除退化通道                      | —               | —               | −0.001（无影响；oMAE −0.023）  |
| V4 persistence-skip 头          | —               | —               | **−0.419±0.07（同向，−25%）** |

诚实解读：

1. **跨传感器图分支有用，但"动态"没有可测量的增益**：去掉图（none / self_only）在 FD002 与 MetroPT 上三 seed 同向恶化（MetroPT +0.22，约 4σ）；而 dynamic vs static 在三个数据集上全在噪声内（FD004 static 还略好）。论文应把贡献归于"跨传感器图结构"，不宜声称"动态图优于静态图"。
2. **gated vs add 无差别**（三数据集 Δ 均 <1σ）。门控的价值不在点预测精度；其暴露的 gate 权重留给第四章作为条件信号另行论证。
3. FD004 上任何结构消融都不显著（噪声 ±0.8）：FD004 的点预测已接近该协议下的可挖掘上限。
4. `self_only` 与 `no_graph` 输出按设计完全一致（`mtkdg.py` 注释），表中只保留一个。

## 4. MC-Graph 门实验（A0 / G1–G4，3 seeds）：干净 test 与损坏输入稳健性

> **口径注**：MetroPT 列为 direct 头批次（2026-10-20）；论文 MetroPT 稳健性采用 §2A.2 的 delta 口径。FD 两列不受影响。

稳健性 = test 上相对 clean RMSE 的中位相对增幅（rate 0.3，3 个 corruption seed），越小越稳健。

| 数据集     | 配置                     | clean RMSE  | Δclean vs A0     | channel_dropout@0.3 | time_block@0.3 |
| ------- | ---------------------- | ----------- | ---------------- | ------------------- | -------------- |
| FD002   | A0                     | 24.75±0.29  | —                | +12.6%              | +20.5%         |
| FD002   | G1 robust-train        | 24.63±0.37  | −0.12（噪声内）       | +3.8%               | +13.6%         |
| FD002   | G2 robust+consistency  | 26.12±0.24  | **+1.37（同向，代价）** | +2.3%               | **+3.0%**      |
| FD002   | G3 availability+robust | 24.59±0.16  | −0.17（噪声内）       | +5.0%               | +11.8%         |
| FD002   | G4 完整 MC-Graph         | 26.25±0.08  | **+1.50（同向，代价）** | +2.1%               | **+3.0%**      |
| FD004   | A0                     | 26.50±0.81  | —                | +17.5%              | +24.8%         |
| FD004   | G1                     | 26.35±0.43  | −0.15（噪声内）       | +0.9%               | +9.8%          |
| FD004   | G2                     | 26.42±0.18  | −0.08（噪声内）       | +1.7%               | **+4.9%**      |
| FD004   | G3                     | 26.40±0.32  | −0.10（噪声内）       | +1.8%               | +11.2%         |
| FD004   | G4                     | 26.40±0.15  | −0.10（噪声内）       | +1.6%               | **+4.5%**      |
| MetroPT | A0                     | 1.690±0.058 | —                | +31.3%              | +5.7%          |
| MetroPT | G1                     | 1.678±0.092 | −0.013（噪声内）      | +21.1%              | +1.7%          |
| MetroPT | G2                     | 1.681±0.092 | −0.009（噪声内）      | +20.9%              | +1.7%          |
| MetroPT | G3                     | 1.694±0.049 | +0.004（噪声内）      | +19.0%              | +1.1%          |
| MetroPT | G4                     | 1.697±0.046 | +0.007（噪声内）      | +18.8%              | +1.1%          |

结论：

1. **稳健性大幅提升**：完整 MC-Graph 把 FD 的 time_block@0.3 退化从 +20~~25% 压到 +3~~5%（约 6×），channel_dropout 从 +13~18% 压到 +2%；MetroPT 的 channel_dropout 从 +31% 降到 +19%。
2. **consistency 项是稳健性的主要来源**（G2 vs G1：FD002 13.6%→3.0%，FD004 9.8%→4.9%），但它在 **FD002 上有干净性能代价 +1.4~1.5 RMSE（约 6%，三 seed 同向）**；FD004 与 MetroPT 无代价。论文须同时报告代价与收益。
3. **availability 编码（G3 vs G1、G4 vs G2）增益很小**（≤2 个百分点，不稳定），可作为次要组件如实描述。
4. 干净性能上 G1/G3 与 A0 完全持平，即"仅 robust-train"是零代价的稳健性改进（FD time_block 退化约减半）。

## 5. 口径备注

1. 全部 run：engine-level / 首月划分、train-only 归一化、validation-best 选型、恢复后一次性 test；artifact 含 manifest（scaler identity + git SHA），位于远程 `results/{dataset}/{run_id}/`（本地同步了除 `.pt/.npz` 外的全部文件）。
2. 全部 3-seed run（20261020 / 20261026 / 20261027_extra 三批）均在**同一 RTX 3090 实例**上产出；批间以 fd002 MTK-DG seed2026 对照验证逐位一致（§1 表头注）。与更早原机（4060 Ti）的验证：MTK-DG 与 TCN 的 seed 2026 跨机差 ≤0.02；**GRU 跨 GPU 差约 1 RMSE**（cuDNN RNN 非确定性），故原机数值（含 §6 附录单 seed 表）不与本页混用。MetroPT §2A delta 批与 direct 批虽同实例，但因输出形式协议修订，仍批内配对、不混表。
3. NASA 的 seed 方差极大（GRU FD004 ±6,922），只作参考并列报告，不用于排名。
4. PatchTST（H=50 下 7-patch）与 TimesNet（top-1 周期单 block）为统一协议下的简化实现，结论限定于该范围。
5. 稳健性评估的 corruption 协议（channel_dropout / time_block，rate 0.1/0.2/0.3，3 corruption seeds）与训练期 robust-train 的 corruption 同源但 seed 独立，详见 `scripts/eval_robustness.py`。channel_dropout 整通道置零同时置零掩码，对 delta 头额外摧毁持久性锚 x_last（§2A.2 解读 1）。

## 6. 附录：旧版单 seed 2026 六模型表（原机 4060 Ti，仅供参考）

FD002：MTK-DG 16.43/24.88/4600 · GRU 20.24/30.96/23413 · TCN 20.75/30.60/23278 · PatchTST 21.78/33.81/80942 · iTransformer 26.47/36.62/58320 · TimesNet 25.06/37.07/126285  
FD004：MTK-DG 18.81/27.13/5311 · GRU 19.26/29.12/15310 · PatchTST 20.64/31.55/96634 · TCN 22.27/30.25/8082 · TimesNet 25.25/34.62/28316 · iTransformer 28.56/36.48/21604  
MetroPT（direct 头，oMAE/oRMSE）：TCN 0.5903/1.9621 · PatchTST 0.6530/1.8552 · MTK-DG 0.6123/1.6266 · iTransformer 0.8415/2.0486 · GRU 0.7896/2.5494 · TimesNet 1.2994/3.6931 —— 注意全部劣于 persistence 的 oMAE 0.4825。
