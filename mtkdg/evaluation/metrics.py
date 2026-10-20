"""点预测指标（全部在原始单位上计算）。"""

from __future__ import annotations

from typing import Dict

import numpy as np

EVALUATOR_VERSION = "ch3_point_evaluator_v1"


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-MAPSS NASA score：d<0 时 exp(-d/13)-1，d>=0 时 exp(d/10)-1。"""
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(score.sum())


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """scalar RUL 指标（raw cycle）。y_true/y_pred 形状 [N] 或 [N,1]。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    err = y_pred - y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "NASA": nasa_score(y_true, y_pred),
        "Bias": float(err.mean()),
    }


def per_engine_rul_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, engine_ids: np.ndarray
):
    """逐 engine 残差（FD002/FD004 test，每个 engine 一个窗口）。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    engine_ids = np.asarray(engine_ids).reshape(-1)
    rows = []
    for eng in np.unique(engine_ids):
        m = engine_ids == eng
        err = float(y_pred[m][0] - y_true[m][0])
        rows.append(
            {
                "engine_id": int(eng),
                "y_true": float(y_true[m][0]),
                "y_pred": float(y_pred[m][0]),
                "residual": err,
                "abs_err": abs(err),
            }
        )
    return rows


def state_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MetroPT 状态预测 overall 指标（raw unit，覆盖 [N,P,C] 全部元素）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    err = y_pred - y_true
    return {
        "overall_MAE": float(np.abs(err).mean()),
        "overall_RMSE": float(np.sqrt((err ** 2).mean())),
        "Bias": float(err.mean()),
    }


def per_sensor_state_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sensor_names
):
    """逐传感器 MAE/RMSE（raw unit），y 形状 [N,P,C]。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rows = []
    for c, name in enumerate(sensor_names):
        err = y_pred[..., c] - y_true[..., c]
        rows.append(
            {
                "sensor": name,
                "channel_index": c,
                "MAE": float(np.abs(err).mean()),
                "RMSE": float(np.sqrt((err ** 2).mean())),
            }
        )
    return rows


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """逐 horizon 误差（raw unit），y 形状 [N,P,C]。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {}
    for p in range(y_true.shape[1]):
        err = y_pred[:, p, :] - y_true[:, p, :]
        out[int(p)] = {
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
        }
    return out
