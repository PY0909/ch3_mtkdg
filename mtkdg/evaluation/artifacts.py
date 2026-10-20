"""run artifact 完整写出。

每个 run 目录（results/{dataset}/{run_id}/）包含：
  config.json / manifest.json / checkpoint_best.pt / checkpoint_last.pt /
  history.json / training_log.csv / training_curves.png /
  validation_metrics.json /
  test_metrics.json / predictions_test.npz / targets_test.npz  （仅 evaluate-test）
  per_engine_metrics.csv + rul_prediction.png  （FD002/FD004）
  per_sensor_metrics.csv + state_prediction.png（MetroPT）
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..models.mtkdg import MODEL_VERSION  # noqa: E402
from .metrics import (  # noqa: E402
    EVALUATOR_VERSION,
    horizon_metrics,
    per_engine_rul_metrics,
    per_sensor_state_metrics,
)


def _sanitize(value):
    """把 NaN/inf 转为 None，保证 JSON 严格合法。"""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2, ensure_ascii=False, default=str)


def _file_sha16(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_identity() -> Dict[str, Optional[str]]:
    """运行代码的 git 身份（评审 W6：可追溯到底哪个 commit 产生该 run）。"""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"code_git_sha": sha, "code_git_dirty": bool(dirty)}
    except Exception:
        return {"code_git_sha": None, "code_git_dirty": None}


def _hardware_info(runner) -> Dict[str, Any]:
    import torch

    info = {
        "device": str(runner.device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    if runner.device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(runner.device)
    return info


def _model_version(runner) -> str:
    head = getattr(runner.model, "head_version", "unknown")
    enc = getattr(runner.model, "model_version", MODEL_VERSION)
    if enc.startswith(MODEL_VERSION):
        return f"{enc}+{head}"
    return f"{enc}+{head}"


def _best_val_metrics(runner) -> Dict[str, Optional[float]]:
    hist = runner.history
    epoch = runner.best_epoch
    try:
        idx = hist["epoch"].index(epoch)
    except ValueError:
        idx = len(hist["epoch"]) - 1
    nasa = hist["val_NASA"][idx] if idx < len(hist["val_NASA"]) else None
    raw_diag = hist.get("val_RMSE_rawtarget")
    out = {
        "epoch": epoch,
        "val_MAE": hist["val_MAE"][idx],
        "val_RMSE": hist["val_RMSE"][idx],
        # 协议脚注（评审 W2）：FD 的 val_RMSE 对 capped 目标计算，与 raw test 指标不可比；
        # val_RMSE_rawtarget 为对未截断目标的监控诊断，不参与 checkpoint 选择。
        "val_RMSE_target": "capped(raw cycle)" if runner.bundle.task == "rul_regression" else "raw",
        # MetroPT 状态预测无 NASA 分，写 null 而不是 NaN
        "val_NASA": nasa if (nasa is not None and nasa == nasa) else None,
    }
    if raw_diag is not None and idx < len(raw_diag) and raw_diag[idx] == raw_diag[idx]:
        out["val_RMSE_rawtarget"] = raw_diag[idx]
    return out


def _plot_training_curves(runner, run_dir: str) -> None:
    hist = runner.history
    epochs = hist["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(epochs, hist["train_loss"], "b-o", ms=3, label="train_loss")
    ax.set_title("(A) Train Loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, hist["val_RMSE"], "r-o", ms=3, label="val_RMSE")
    ax.plot(epochs, hist["val_MAE"], "g--s", ms=3, label="val_MAE")
    ax.axvline(runner.best_epoch, color="k", ls=":", label=f"best={runner.best_epoch}")
    ax.set_title("(B) Validation Metrics (raw unit)")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    nasa = [v for v in hist["val_NASA"]]
    if any(v == v for v in nasa):  # 非 NaN 才画（RUL 任务）
        ax.plot(epochs, nasa, "m-o", ms=3, label="val_NASA")
        ax.set_yscale("log")
        ax.set_title("(C) Validation NASA Score")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "N/A (state forecast task)", ha="center", va="center")
        ax.set_title("(C) Validation NASA Score")

    ax = axes[1, 1]
    ax.plot(epochs, hist["val_RMSE"], "r-", label="val_RMSE (zoom)")
    ax.set_title("(D) Validation RMSE")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _plot_rul_prediction(preds: np.ndarray, trues: np.ndarray, run_dir: str) -> None:
    preds = preds.reshape(-1)
    trues = trues.reshape(-1)
    order = np.argsort(trues)
    limit = min(300, len(order))
    idx = order[:limit]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trues[idx], "k-", lw=1.5, label="True RUL")
    ax.plot(preds[idx], "r--", lw=1.5, label="Pred RUL")
    ax.set_title(f"RUL Prediction (first {limit} engines sorted by true RUL)")
    ax.set_xlabel("engine index (sorted)")
    ax.set_ylabel("RUL (cycles)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "rul_prediction.png"), dpi=150)
    plt.close(fig)


def _plot_state_prediction(preds: np.ndarray, trues: np.ndarray,
                           sensor_names, run_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    p = preds[0]
    t = trues[0]
    half = p.shape[0] // 2
    ax.plot(range(half), t[:half, 0], "k-", label="true (1st half horizon)")
    ax.plot(range(half, p.shape[0]), p[half:, 0], "r--", label="pred")
    ax.plot(range(half, p.shape[0]), t[half:, 0], "g:", label="true")
    ax.set_title(f"State prediction example: {sensor_names[0]} (test sample 0)")
    ax.set_xlabel("horizon step")
    ax.set_ylabel(sensor_names[0])
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    rows = per_sensor_state_metrics(trues, preds, sensor_names)
    names = [r["sensor"] for r in rows]
    maes = [r["MAE"] for r in rows]
    ax.bar(range(len(names)), maes, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Per-sensor MAE (raw unit)")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "state_prediction.png"), dpi=150)
    plt.close(fig)


def _write_training_log_csv(runner, run_dir: str) -> None:
    hist = runner.history
    keys = [k for k in ("epoch", "train_loss", "lr", "val_MAE", "val_RMSE",
                        "val_NASA", "val_RMSE_rawtarget") if k in hist and hist[k]]
    lines = [",".join(keys)]
    for row in zip(*[hist[k] for k in keys]):
        lines.append(",".join(_fmt(v) for v in row))
    with open(os.path.join(run_dir, "training_log.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _per_month_state_metrics(preds, trues, window_months):
    """MetroPT test 逐月 MAE/RMSE（raw unit，评审 W4：把漂移归因变成曲线）。"""
    months = list(window_months)
    out = {}
    for m in sorted(set(months)):
        idx = [i for i, x in enumerate(months) if x == m]
        err = preds[idx] - trues[idx]
        out[m] = {
            "n_windows": len(idx),
            "overall_MAE": float(np.abs(err).mean()),
            "overall_RMSE": float(np.sqrt((err ** 2).mean())),
        }
    return out


def save_run_artifacts(runner, bundle, fit_info: Dict, test_result: Optional[Dict]) -> None:
    run_dir = runner.run_dir
    os.makedirs(run_dir, exist_ok=True)
    task = bundle.task

    # 1) config / history（config 覆盖写入以含最终字段）
    _write_json(os.path.join(run_dir, "config.json"), runner.cfg.to_dict())
    _write_json(
        os.path.join(run_dir, "history.json"),
        {
            "history": runner.history,
            "best_epoch": runner.best_epoch,
            "best_val_RMSE": runner.best_metric,
            "epochs_ran": runner.epochs_ran,
            "fit_info": fit_info,
        },
    )

    # 2) 逐 epoch CSV + 训练曲线
    _write_training_log_csv(runner, run_dir)
    _plot_training_curves(runner, run_dir)

    # 3) validation 指标（best epoch）
    _write_json(os.path.join(run_dir, "validation_metrics.json"), {
        "best": _best_val_metrics(runner),
        "selection_metric": "val_RMSE",
        "n_val_samples": bundle.val.n_samples,
    })

    # 4) manifest（可追溯性）
    best_path = os.path.join(run_dir, "checkpoint_best.pt")
    manifest = {
        "run_id": runner.cfg.run_id,
        "dataset": bundle.dataset_name,
        "task": task,
        "model_version": _model_version(runner),
        "seed": runner.cfg.seed,
        "split_seed": runner.cfg.split_seed,
        "split_identity": bundle.split_identity,
        "split_identity_hash": bundle.identity,
        "feature_schema": bundle.feature_schema,
        "target_schema": bundle.target_schema,
        "normalizer_identity": {
            "feature": bundle.feature_scaler.identity(),
            "target": bundle.target_scaler.identity(),
            "feature_state": bundle.feature_scaler.state_dict(),
            "target_state": bundle.target_scaler.state_dict(),
        },
        "checkpoint_identity": {
            "best_sha256_16": _file_sha16(best_path),
            "best_epoch": runner.best_epoch,
            "best_val_RMSE": runner.best_metric,
        },
        "evaluator_version": EVALUATOR_VERSION,
        "sample_counts": bundle.sample_counts(),
        "hardware": _hardware_info(runner),
        "training_duration_s": round(runner.train_duration_s, 3),
        "inference_duration_s": round(runner.inference_duration_s, 3),
        "parameter_count": runner.model.count_parameters(),
        "epochs_ran": runner.epochs_ran,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **_git_identity(),
        # 大数组/长列表（如 window_months）不进 manifest，逐项见各 split 产物
        "protocol_extra": {
            k: v for k, v in bundle.meta.items()
            if not isinstance(v, np.ndarray)
            and not (isinstance(v, (list, tuple)) and len(v) > 64)
        },
    }
    _write_json(os.path.join(run_dir, "manifest.json"), manifest)

    # 5) test artifacts（只有显式 evaluate-test 才有）
    if test_result is not None:
        preds = np.asarray(test_result["preds_raw"], dtype=np.float32)
        trues = np.asarray(test_result["trues_raw"], dtype=np.float32)
        np.savez_compressed(os.path.join(run_dir, "predictions_test.npz"), predictions=preds)
        np.savez_compressed(os.path.join(run_dir, "targets_test.npz"), targets=trues)

        test_payload: Dict[str, Any] = {
            "metrics": test_result["metrics"],
            "n_test_samples": int(trues.shape[0]),
            "evaluated_once": True,
            "checkpoint": "validation-best",
        }
        if task == "rul_regression":
            rows = per_engine_rul_metrics(
                trues, preds, bundle.test.meta["engine_ids"]
            )
            with open(os.path.join(run_dir, "per_engine_metrics.csv"), "w",
                      encoding="utf-8") as f:
                f.write("engine_id,y_true,y_pred,residual,abs_err\n")
                for r in rows:
                    f.write(
                        f"{r['engine_id']},{r['y_true']:.4f},{r['y_pred']:.4f},"
                        f"{r['residual']:.4f},{r['abs_err']:.4f}\n"
                    )
            _plot_rul_prediction(preds, trues, run_dir)
        else:
            names = bundle.feature_schema["channels"]
            rows = per_sensor_state_metrics(trues, preds, names)
            # 归一化空间 per-sensor 列（评审 W3：退化通道的影响在原始单位下不可见）
            preds_n = bundle.target_scaler.transform(preds)
            trues_n = bundle.target_scaler.transform(trues)
            rows_n = per_sensor_state_metrics(trues_n, preds_n, names)
            with open(os.path.join(run_dir, "per_sensor_metrics.csv"), "w",
                      encoding="utf-8") as f:
                f.write("sensor,channel_index,MAE,RMSE,MAE_norm,RMSE_norm\n")
                for r, rn in zip(rows, rows_n):
                    f.write(
                        f"{r['sensor']},{r['channel_index']},{r['MAE']:.6f},{r['RMSE']:.6f},"
                        f"{rn['MAE']:.6f},{rn['RMSE']:.6f}\n"
                    )
            _plot_state_prediction(preds, trues, names, run_dir)
            test_payload["horizon_metrics"] = {
                str(k): v for k, v in horizon_metrics(trues, preds).items()
            }
            window_months = bundle.test.meta.get("window_months")
            if window_months is not None:
                per_month = _per_month_state_metrics(preds, trues, window_months)
                test_payload["per_month_metrics"] = per_month
                with open(os.path.join(run_dir, "per_month_metrics.csv"), "w",
                          encoding="utf-8") as f:
                    f.write("month,n_windows,overall_MAE,overall_RMSE\n")
                    for month, m in per_month.items():
                        f.write(f"{month},{m['n_windows']},{m['overall_MAE']:.6f},"
                                f"{m['overall_RMSE']:.6f}\n")
        _write_json(os.path.join(run_dir, "test_metrics.json"), test_payload)
