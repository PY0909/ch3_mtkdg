---
type: experiment
node_id: exp:ch3-matrix-fd002
title: "FD002 统一协议单 seed 矩阵（6 模型，seed 2026，cosine）"
idea_id: "idea:mtkdg-encoder"
verdict: partial
confidence: medium
date: ""
hardware: "RTX 4060 Ti, torch 2.5.1"
duration: "~12 min/run"
provenance: "results/fd002/fd002_*_seed2026_r2_20261013_{2115,2224}00/ (manifest+test_metrics+per_engine CSV)"
added: 2026-10-14T04:45:32Z
tags: []
---

# FD002 统一协议单 seed 矩阵（6 模型，seed 2026，cosine）

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:mtkdg-encoder`

## Metrics
MTK-DG(r2): MAE 22.20 RMSE 30.66 NASA 11114 | TCN 26.18/35.24/51961 | GRU 27.88/36.35/44057 | TimesNet 30.34/40.02/68331 | iTransformer 34.57/43.73/92903 | PatchTST 35.03/44.94/257773

## Reasoning
MTK-DG 三指标全部第 1 且参数量最小(40.9k)；单 seed、无配对检验，不得作优劣结论

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

