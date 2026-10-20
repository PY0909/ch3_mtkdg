"""R2 kill-test pilot（零训练诊断）：冻结 V1 checkpoint 在评估期 corruption 下的自然退化。

裁决门（预注册，先于任何结果写定）：
- R2（鲁棒线）值得继续 ⟺ 在 channel-dropout 0.3 或 time-block 0.3 下，
  ≥2/3 数据集的相对 RMSE 增幅中位数 ≥ 10%。
- 未达标 ⟺ 基线已天然鲁棒，该方向无问题可解，回 R1 概率主线。

corruption 只作用于传感器通道（FD 保留 3 个工况通道与 condition 输入；
MetroPT 全部 15 通道）。评估期施加、不重训、不调参；结果写入
idea-stage/PILOT_corruption_degradation.json（诊断产物，不进 results/ 主协议目录）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtkdg.config import ExperimentConfig, MTKDGConfig  # noqa: E402
from mtkdg.data.bundle import build_bundle  # noqa: E402
from mtkdg.models.heads import MTKDGPointModel  # noqa: E402
from mtkdg.evaluation.metrics import rul_metrics, state_metrics  # noqa: E402
from mtkdg.training.checkpoint import load_checkpoint  # noqa: E402

RUNS = {
    "fd002": "results/fd002/fd002_mtkdg_v1_percond_20261014_loop2",
    "fd004": "results/fd004/fd004_mtkdg_v1_percond_20261014_loop2",
    "metropt3": "results/metropt3/metropt3_mtkdg_seed2026_r2_20261013_211500",
}
RATES = (0.1, 0.2, 0.3)
CORRUPT_SEEDS = (0, 1, 2)
GATE = {"min_rel_increase": 0.10, "min_datasets": 2}


def load_cfg(run_dir: str) -> ExperimentConfig:
    d = json.load(open(os.path.join(run_dir, "config.json")))
    fields = {f for f in ExperimentConfig.__dataclass_fields__ if f in d}
    kw = {k: d[k] for k in fields}
    if kw.get("pred_len") is None:
        kw["pred_len"] = 0
    kw["model"] = MTKDGConfig(**d["model"])
    kw["evaluate_test"] = False
    return ExperimentConfig(**kw)


def channel_dropout(x, mask, rate, sensor_slice, rng):
    B, H, C = x.shape
    m = mask.clone()
    x = x.clone()
    drop = torch.from_numpy(rng.random((B, C)) < rate).to(x.device)
    drop[:, :sensor_slice.start] = False          # 保护工况通道（FD 前 3 列）
    drop = drop.unsqueeze(1).float()              # [B,1,C]
    m = m * (1.0 - drop)
    x = x * (1.0 - drop)
    return x, m


def time_block_mask(x, mask, ratio, sensor_slice, rng):
    B, H, C = x.shape
    m = mask.clone()
    x = x.clone()
    blen = max(1, int(round(H * ratio)))
    starts = torch.from_numpy(rng.integers(0, H - blen + 1, size=B)).to(x.device)
    rows = torch.arange(H, device=x.device).unsqueeze(0)  # [1,H]
    keep = ~((rows >= starts.unsqueeze(1)) & (rows < (starts.unsqueeze(1) + blen)))  # [B,H]
    keep = keep.unsqueeze(-1).float()             # [B,H,1]
    m = m * keep
    x = x * keep
    return x, m


@torch.no_grad()
def evaluate(model, split, bundle, device, batch_size=256,
             corrupt=None, sensor_slice=slice(0, None), seed=0):
    model.eval()
    preds, trues = [], []
    n = split.n_samples
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        x = torch.from_numpy(split.x[idx]).to(device)
        t = torch.from_numpy(split.t[idx]).to(device)
        m = torch.from_numpy(split.mask[idx]).to(device)
        if corrupt is not None:
            rng = np.random.default_rng(seed * 100_003 + start)
            x, m = corrupt(x, m, sensor_slice=sensor_slice, rng=rng)
        batch = {"x": x, "t": t, "mask": m}
        if split.condition is not None:
            batch["condition"] = torch.from_numpy(split.condition[idx]).to(device)
        out = model(batch["x"], batch["t"], batch["mask"], batch.get("condition"))
        if bundle.task == "rul_regression":
            preds.append(out.cpu().numpy())
        else:
            preds.append(bundle.target_scaler.inverse_transform(
                out.cpu().numpy().reshape(-1, bundle.n_channels)
            ).reshape(out.shape[0], -1, bundle.n_channels))
    pred = np.concatenate(preds, axis=0)
    true = split.y_raw
    if bundle.task == "rul_regression":
        met = rul_metrics(true.reshape(-1), pred.reshape(-1))
        return {"MAE": met["MAE"], "RMSE": met["RMSE"]}
    met = state_metrics(true, pred)
    return {"MAE": met["overall_MAE"], "RMSE": met["overall_RMSE"]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"gate": GATE, "datasets": {}}
    print(f"pilot on {device} | gate: rel_RMSE +{GATE['min_rel_increase']:.0%} "
          f"at rate 0.3 on >= {GATE['min_datasets']}/3 datasets")

    for ds, run_dir in RUNS.items():
        if not os.path.exists(run_dir):
            print(f"skip {ds}: {run_dir} missing")
            continue
        cfg = load_cfg(run_dir)
        cfg.output_dir = "/tmp/pilot_unused"
        bundle = build_bundle(cfg)
        pred_len = bundle.train.y.shape[1] if bundle.task == "state_forecast" else cfg.pred_len
        model = MTKDGPointModel(
            task=bundle.task, n_channels=bundle.n_channels, pred_len=pred_len,
            model_cfg=cfg.model, rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
        ).to(device)
        load_checkpoint(os.path.join(run_dir, "checkpoint_best.pt"), model=model)
        sensor_slice = slice(3, None) if bundle.task == "rul_regression" else slice(0, None)
        split = bundle.test

        clean = evaluate(model, split, bundle, device, sensor_slice=sensor_slice)
        entry = {"clean": clean, "corruptions": {}}
        for ctype, fn in (("channel_dropout", channel_dropout), ("time_block", time_block_mask)):
            for rate in RATES:
                rel = []
                for cs in CORRUPT_SEEDS:
                    bound = lambda x, m, sensor_slice, rng, fn=fn, rate=rate: fn(
                        x, m, rate, sensor_slice, rng)
                    met = evaluate(model, split, bundle, device, corrupt=bound,
                                   sensor_slice=sensor_slice, seed=cs)
                    rel.append((met["RMSE"] - clean["RMSE"]) / clean["RMSE"])
                entry["corruptions"][f"{ctype}_{rate}"] = {
                    "rel_RMSE_increase_median": float(np.median(rel)),
                    "rel_RMSE_increase_all": [float(v) for v in rel],
                }
        report["datasets"][ds] = entry
        print(f"== {ds} clean RMSE={clean['RMSE']:.4f}")
        for k, v in entry["corruptions"].items():
            if k.endswith("0.3") or k.endswith("0.1"):
                print(f"   {k:22s} rel_RMSE +{v['rel_RMSE_increase_median']:+.1%}")

    # 裁决
    passed = []
    for ds, entry in report["datasets"].items():
        worst = max(v["rel_RMSE_increase_median"]
                    for k, v in entry["corruptions"].items() if k.endswith("0.3"))
        entry["worst_rel_increase_at_0.3"] = worst
        if worst >= GATE["min_rel_increase"]:
            passed.append(ds)
    report["gate_result"] = {
        "datasets_passing": passed,
        "verdict": "R2_VIABLE" if len(passed) >= GATE["min_datasets"] else "R2_KILL_back_to_R1",
    }
    print("\nGATE:", report["gate_result"]["verdict"], "| passing:", passed)
    out = "idea-stage/PILOT_corruption_degradation.json"
    json.dump(report, open(out, "w"), indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
