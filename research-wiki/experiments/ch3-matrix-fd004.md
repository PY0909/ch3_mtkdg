---
type: experiment
node_id: exp:ch3-matrix-fd004
title: "FD004 统一协议单 seed 矩阵（6 模型，seed 2026，cosine）"
idea_id: "idea:mtkdg-encoder"
verdict: partial
confidence: medium
date: ""
hardware: "RTX 4060 Ti, torch 2.5.1"
duration: "~14 min/run"
provenance: "results/fd004/fd004_*_seed2026_r2_20261013_{2115,2224}00/"
added: 2026-10-14T04:45:32Z
tags: []
---

# FD004 统一协议单 seed 矩阵（6 模型，seed 2026，cosine）

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:mtkdg-encoder`

## Metrics
MTK-DG(r2): MAE 26.38 RMSE 36.63 NASA 236567 | TCN 30.43/38.66/51590 | TimesNet 34.51/42.92/74164 | GRU 35.75/45.07/218913 | PatchTST 38.33/47.86/227320 | iTransformer 40.03/50.21/1979609

## Reasoning
MAE/RMSE 第 1 但 NASA 第 4：分解显示 85% NASA 来自单引擎(true 184→pred 25, capped 目标结构性低估, G4)；99% NASA 来自 67 个 true>125 引擎；去单引擎后 35k<TCN 51k 但属事后选择不可引用

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

