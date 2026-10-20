---
type: experiment
node_id: exp:v1-unified-matrix
title: "V1 分工况归一化统一重跑（6 模型 × FD002/FD004，seed 2026）"
idea_id: "idea:mtkdg-encoder"
verdict: partial
confidence: medium
date: ""
hardware: "RTX 4060 Ti"
duration: "~100 min/12 runs"
provenance: "results/*/*v1*percond*_20261014_loop2/ + refine-logs/v1_matrix_fd.json"
added: 2026-10-14T08:57:21Z
tags: []
---

# V1 分工况归一化统一重跑（6 模型 × FD002/FD004，seed 2026）

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:mtkdg-encoder`

## Metrics
FD002: mtkdg 16.43/24.88/4600 全指标第1(ΔRMSE -5.73 vs ±0.42); FD004: mtkdg 18.81/27.13/5311 全指标第1 但 MAE vs GRU 噪声内(-0.45 vs ±0.74); NASA 结构性修复 236567→5311(-98%); 全模型受益(tcn RMSE -13/-22%)

## Reasoning
V1 升级主协议成立(48× 工况方差证据兑现); 验证了统一重跑必要性(baseline 同吃红利否则 mtkdg 优势被高估); 收敛提前(best_ep 11-137); FD004 GRU 并列待多 seed 裁决

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

