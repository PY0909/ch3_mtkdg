"""鲁棒性评估：对任意 run 目录计算 clean + corruption 曲线（复用预注册 corruption）。

用法：python scripts/eval_robustness.py <run_dir> [--out <json>]
输出：clean MAE/RMSE + 各 corruption 配置的中位相对退化（3 个 corruption seeds）。
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
from mtkdg.evaluation.metrics import rul_metrics, state_metrics  # noqa: E402
from mtkdg.models.heads import MTKDGPointModel  # noqa: E402
from mtkdg.training.checkpoint import load_checkpoint  # noqa: E402
from mtkdg.training.corruption import channel_dropout, time_block_mask  # noqa: E402

RATES = (0.1, 0.2, 0.3)
CORRUPT_SEEDS = (0, 1, 2)


def load_cfg(run_dir: str) -> ExperimentConfig:
    d = json.load(open(os.path.join(run_dir, "config.json")))
    fields = {f for f in ExperimentConfig.__dataclass_fields__ if f in d}
    kw = {k: d[k] for k in fields}
    if kw.get("pred_len") is None:
        kw["pred_len"] = 0
    kw["model"] = MTKDGConfig(**d["model"])
    kw["evaluate_test"] = False
    return ExperimentConfig(**kw)


@torch.no_grad()
def evaluate(model, split, bundle, device, corrupt=None, protect=0, seed=0,
             batch_size=256):
    model.eval()
    preds = []
    for start in range(0, split.n_samples, batch_size):
        idx = np.arange(start, min(start + batch_size, split.n_samples))
        x = torch.from_numpy(split.x[idx]).to(device)
        t = torch.from_numpy(split.t[idx]).to(device)
        m = torch.from_numpy(split.mask[idx]).to(device)
        if corrupt is not None:
            x, m = corrupt(x, m)
        batch_cond = torch.from_numpy(split.condition[idx]).to(device) \
            if split.condition is not None else None
        out = model(x, t, m, batch_cond)
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
    else:
        met = state_metrics(true, pred)
    return {"MAE": met.get("MAE", met.get("overall_MAE")),
            "RMSE": met.get("RMSE", met.get("overall_RMSE"))}


def make_corrupt(fn, rate, protect, seed):
    """确定性的评估期 corruption 闭包（CPU generator，逐 batch 顺序采样）。"""
    gen = torch.Generator()
    gen.manual_seed(seed * 100_003 + 17)

    def _c(x, m):
        return fn(x, m, rate, protect_channels=protect, generator=gen)
    return _c


def main():
    run_dir = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        os.path.join(run_dir, "robustness_eval.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(run_dir)
    cfg.output_dir = "/tmp/eval_unused"
    bundle = build_bundle(cfg)
    pred_len = bundle.train.y.shape[1] if bundle.task == "state_forecast" else cfg.pred_len
    model = MTKDGPointModel(
        task=bundle.task, n_channels=bundle.n_channels, pred_len=pred_len,
        model_cfg=cfg.model, rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
    ).to(device)
    load_checkpoint(os.path.join(run_dir, "checkpoint_best.pt"), model=model)
    protect = bundle.condition_dim
    split = bundle.test

    clean = evaluate(model, split, bundle, device)
    entry = {"clean": clean, "corruptions": {}}
    for ctype, fn in (("channel_dropout", channel_dropout), ("time_block", time_block_mask)):
        for rate in RATES:
            rel = []
            for cs in CORRUPT_SEEDS:
                met = evaluate(model, split, bundle, device,
                               corrupt=make_corrupt(fn, rate, protect, cs),
                               protect=protect, seed=cs)
                rel.append((met["RMSE"] - clean["RMSE"]) / clean["RMSE"])
            entry["corruptions"][f"{ctype}_{rate}"] = {
                "rel_RMSE_increase_median": float(np.median(rel)),
                "rel_RMSE_increase_all": [float(v) for v in rel],
            }
    entry["worst_rel_increase_at_0.3"] = max(
        v["rel_RMSE_increase_median"] for k, v in entry["corruptions"].items()
        if k.endswith("0.3"))
    json.dump(entry, open(out_path, "w"), indent=2)
    print(f"{os.path.basename(run_dir)}: clean RMSE={clean['RMSE']:.4f} "
          f"worst@0.3={entry['worst_rel_increase_at_0.3']:+.1%} -> {out_path}")


if __name__ == "__main__":
    main()


@torch.no_grad()
def outage_sweep(run_dir: str, device=None):
    """持久断连评估（评审 Round1 第二优先级）：逐一完全屏蔽单个传感器通道
    整窗，报告 mean/worst RMSE 退化。预注册协议：全通道 sweep（不含保护通道）。
    """
    run_dir = os.path.abspath(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(run_dir)
    cfg.output_dir = "/tmp/eval_unused"
    bundle = build_bundle(cfg)
    pred_len = bundle.train.y.shape[1] if bundle.task == "state_forecast" else cfg.pred_len
    model = MTKDGPointModel(
        task=bundle.task, n_channels=bundle.n_channels, pred_len=pred_len,
        model_cfg=cfg.model, rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
    ).to(device)
    load_checkpoint(os.path.join(run_dir, "checkpoint_best.pt"), model=model)
    protect = bundle.condition_dim
    split = bundle.test
    C = bundle.n_channels
    clean = evaluate(model, split, bundle, device)
    per_sensor = {}
    for k in range(protect, C):
        def _outage(x, m, k=k):
            mask = m.clone()
            mask[:, :, k] = 0.0
            return x * mask, mask
        met = evaluate(model, split, bundle, device, corrupt=_outage, protect=protect)
        per_sensor[int(k)] = (met["RMSE"] - clean["RMSE"]) / clean["RMSE"]
    vals = list(per_sensor.values())
    result = {
        "clean_RMSE": clean["RMSE"],
        "mean_rel_increase": float(np.mean(vals)),
        "worst_rel_increase": float(np.max(vals)),
        "worst_channel": int(list(per_sensor.keys())[int(np.argmax(vals))]),
        "per_sensor": per_sensor,
    }
    out = os.path.join(run_dir, "outage_sweep.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"{os.path.basename(run_dir)}: outage mean={result['mean_rel_increase']:+.1%} "
          f"worst={result['worst_rel_increase']:+.1%} (ch {result['worst_channel']})")
    return result


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--outage-sweep":
    outage_sweep(sys.argv[1])
