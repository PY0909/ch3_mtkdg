---
type: experiment
node_id: exp:ch3-matrix-metropt3
title: "MetroPT-3 统一协议单 seed 矩阵（6 模型，首月协议，seed 2026）"
idea_id: "idea:mtkdg-encoder"
verdict: partial
confidence: medium
date: ""
hardware: "RTX 4060 Ti, torch 2.5.1"
duration: "~1-3 min/run"
provenance: "results/metropt3/metropt3_*_seed2026_r2_20261013_{2115,2224}00/"
added: 2026-10-14T04:45:32Z
tags: []
---

# MetroPT-3 统一协议单 seed 矩阵（6 模型，首月协议，seed 2026）

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:mtkdg-encoder`

## Metrics
MTK-DG(r2): oMAE 0.612 oRMSE 1.627 | TCN 0.590/1.962 | PatchTST 0.653/1.855 | iTransformer 0.842/2.049 | GRU 0.790/2.549 | TimesNet 1.299/3.693 (raw unit, 21699 test 窗口)

## Reasoning
oRMSE 第 1(明显), oMAE 第 2(差 TCN 0.022, 小于可能 seed 噪声)；逐月 RMSE 1.43-1.81 非单调；退化通道 Oil_level 11.3σ 主导归一化误差(G6)

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

