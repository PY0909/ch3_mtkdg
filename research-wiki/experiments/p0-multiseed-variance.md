---
type: experiment
node_id: exp:p0-multiseed-variance
title: "P0 多 seed 方差基准（mtkdg+tcn × 3 数据集 × seeds 2026/27/28，V0 协议）"
idea_id: "idea:mtkdg-encoder"
verdict: partial
confidence: high
date: ""
hardware: "RTX 4060 Ti"
duration: "~70 min/12 runs"
provenance: "results/*/*seed202{7,8}_20261014_loop2/ + P0 分析"
added: 2026-10-14T08:57:20Z
tags: []
---

# P0 多 seed 方差基准（mtkdg+tcn × 3 数据集 × seeds 2026/27/28，V0 协议）

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:mtkdg-encoder`

## Metrics
MetroPT: mtkdg oRMSE 1.690±0.058 vs tcn 2.101±0.121 (配对Δ=-0.411±0.066 3seeds同向=显著); oMAE 平局(-0.006±0.025 反向). FD002: mtkdg 全指标显著优(ΔMAE -3.86±0.34). FD004: MAE 显著优(-3.20±0.80), RMSE 平局(-0.59±1.52 反向), NASA 全输但方差4.5×(engine125三seed塌到10-25, G4确定性失败模式)

## Reasoning
建立了 seed 噪声基准(FD MAE ±0.5-0.9, MetroPT oMAE ±0.01-0.035); MetroPT 0.022 MAE 差距定性为噪声; NASA 不适合作 FD004 单一排名指标

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

