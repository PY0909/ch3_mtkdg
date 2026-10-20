"""Train-only per-channel standard scaler.

硬约束：归一化统计量只能由 train split 拟合；transform 应用于所有 split；
inverse_transform 用于把标准化空间指标恢复到原始单位。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-8


@dataclass
class TrainOnlyStandardScaler:
    """逐通道 z-score 标准化器（仅 train 拟合）。

    fit 输入形状 [N, C]（N 为样本行数，C 为通道数）；也接受 [N, T, C]，
    统计量在 N、T 两轴上联合计算。
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    n_samples: int = 0
    fitted: bool = False
    name: str = "train_only_standard_v1"
    # 方差下限（pre-registered）：train 内近常数通道的 std 被抬到该值，
    # 防止 test 时真实波动被放大成 10^3 sigma 级特征。0.0 表示不启用。
    min_std: float = 0.0

    def fit(self, x: np.ndarray) -> "TrainOnlyStandardScaler":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            axes = 0
        elif x.ndim == 3:
            axes = (0, 1)
        else:
            raise ValueError(f"expect 2D/3D array, got shape {x.shape}")
        self.mean = x.mean(axis=axes)
        std = np.sqrt(x.var(axis=axes) + _EPS)
        self.std = np.maximum(std, self.min_std) if self.min_std > 0 else std
        self.n_samples = int(x.shape[0])
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("scaler not fitted; call fit() on train split first")
        x = np.asarray(x, dtype=np.float32)
        out = (x.astype(np.float64) - self.mean) / self.std
        return out.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("scaler not fitted")
        x = np.asarray(x, dtype=np.float64)
        out = x * self.std + self.mean
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_samples": self.n_samples,
            "min_std": self.min_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "TrainOnlyStandardScaler":
        obj = cls()
        obj.mean = np.asarray(state["mean"], dtype=np.float64)
        obj.std = np.asarray(state["std"], dtype=np.float64)
        obj.n_samples = int(state["n_samples"])
        obj.fitted = True
        obj.name = state.get("name", cls.name)
        obj.min_std = float(state.get("min_std", 0.0))
        return obj

    def identity(self) -> str:
        """统计量指纹，写入 manifest 以证明归一化来源可追溯。"""
        payload = json.dumps(
            {
                "name": self.name,
                "mean": [round(float(v), 8) for v in self.mean],
                "std": [round(float(v), 8) for v in self.std],
                "n_samples": self.n_samples,
                "min_std": self.min_std,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class PerConditionScaler:
    """分工况归一化（V1 消融，FD002/FD004 六工况）。

    每个工况编码拟合独立 TrainOnlyStandardScaler（仅用该工况的 train-engine 行），
    另拟一个全局 scaler 作为未见工况的回退。工况内近常数通道（如 settings 本身）
    由 min_std 下限保护。接口与 TrainOnlyStandardScaler 对齐（bundle/artifacts 兼容）。
    """

    def __init__(self, min_std: float = 0.01, min_group: int = 30):
        self.min_std = min_std
        # 训练行数不足 min_group 的工况不单独拟 scaler（防小样本组产生
        # 垃圾统计），transform 时这些组回退全局
        self.min_group = min_group
        self.scalers: dict = {}
        self.fallback = TrainOnlyStandardScaler(min_std=min_std)
        self.fitted = False
        self.name = "per_condition_standard_v1"
        self.n_samples = 0

    def fit(self, x: np.ndarray, cond_codes: np.ndarray) -> "PerConditionScaler":
        x = np.asarray(x, dtype=np.float64)
        cond_codes = np.asarray(cond_codes)
        self.fallback.fit(x)
        for code in np.unique(cond_codes):
            rows = x[cond_codes == code]
            if len(rows) < self.min_group:
                continue
            scaler = TrainOnlyStandardScaler(min_std=self.min_std)
            scaler.fit(rows)
            self.scalers[int(code)] = scaler
        self.n_samples = int(x.shape[0])
        self.fitted = True
        return self

    def transform(self, x: np.ndarray, cond_codes: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("per-condition scaler not fitted")
        x = np.asarray(x, dtype=np.float64)
        cond_codes = np.asarray(cond_codes)
        out = self.fallback.transform(x)
        for code, scaler in self.scalers.items():
            m = cond_codes == code
            if m.any():
                out[m] = scaler.transform(x[m])
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "min_std": self.min_std,
            "min_group": self.min_group,
            "n_samples": self.n_samples,
            "fallback": self.fallback.state_dict(),
            "conditions": {
                str(k): v.state_dict() for k, v in self.scalers.items()
            },
        }

    def identity(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "min_std": self.min_std,
                "min_group": self.min_group,
                "fallback": self.fallback.identity(),
                "conditions": {str(k): v.identity() for k, v in self.scalers.items()},
                "n_samples": self.n_samples,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
