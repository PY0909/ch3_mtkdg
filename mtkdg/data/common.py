"""数据协议通用工具：连续窗口切分与身份哈希。

窗口规则（两个数据集共用）：
- 一个样本由 history `[i, i+H)` 与 target `[i+H, i+H+P)` 组成；
- 完整区间 `[i, i+H+P)` 必须落在同一个 split 的连续行范围内，
  任何窗口不允许跨越 split 边界。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SplitArrays:
    """一个 split 的窗口化数组集合（np.float32）。"""

    name: str
    x: np.ndarray                    # [N, H, C] 标准化特征
    t: np.ndarray                    # [N, H] 时间坐标（MetroPT: 相对秒；C-MAPSS: cycle 序号）
    y: np.ndarray                    # 标准化目标：MetroPT [N,P,C]，C-MAPSS [N,1]
    y_raw: np.ndarray                # 原始单位目标（评估用）
    mask: np.ndarray                 # [N, H, C] 观测掩码（主实验全 1）
    starts: np.ndarray               # [N] 每个窗口在本 split 全局行坐标中的起点
    ends: np.ndarray                 # [N] 每个窗口 target 结束行（不含）
    condition: Optional[np.ndarray] = None  # C-MAPSS: [N, H, 3] 标准化工况
    # 仅 C-MAPSS train/val：未截断的 raw RUL（监控诊断用，选择 checkpoint 仍按协议用 capped）
    y_raw_uncapped: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    def summary(self) -> Dict:
        return {
            "split": self.name,
            "n_samples": self.n_samples,
            "x_shape": list(self.x.shape),
            "y_shape": list(self.y.shape),
        }


def build_contiguous_windows(
    values: np.ndarray,
    times: np.ndarray,
    history_len: int,
    pred_len: int,
    stride: int,
    split_name: str,
    row_offset: int,
    extra_channels: Optional[Dict[str, np.ndarray]] = None,
) -> SplitArrays:
    """在一个 split 的连续行范围上切窗口，保证不越界。

    参数
    ----
    values : [T, C] 本 split 的特征行（已标准化）
    times  : [T] 时间坐标
    row_offset : 本 split 第一行在全局表中的行号（用于边界审计）
    extra_channels : 形如 {"condition": [T, 3]} 的附加通道，按同样方式切窗
    """
    values = np.asarray(values, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    total = values.shape[0]
    span = history_len + pred_len
    if total < span:
        return SplitArrays(
            name=split_name,
            x=np.zeros((0, history_len, values.shape[-1]), dtype=np.float32),
            t=np.zeros((0, history_len), dtype=np.float32),
            y=np.zeros((0, pred_len), dtype=np.float32),
            y_raw=np.zeros((0, pred_len), dtype=np.float32),
            mask=np.zeros((0, history_len, values.shape[-1]), dtype=np.float32),
            starts=np.zeros((0,), dtype=np.int64),
            ends=np.zeros((0,), dtype=np.int64),
            meta={"row_offset": row_offset, "n_rows": total, "empty_reason": "rows < H+P"},
        )

    n_channels = values.shape[-1]
    starts_list: List[int] = []
    xs, ts, ys = [], [], []
    extras: Dict[str, List[np.ndarray]] = {k: [] for k in (extra_channels or {})}

    for start in range(0, total - span + 1, stride):
        h_end = start + history_len
        end = start + span
        starts_list.append(start)
        xs.append(values[start:h_end])
        ts.append(times[start:h_end])
        ys.append(values[h_end:end])  # MetroPT: 目标即未来传感器值
        for key, arr in (extra_channels or {}).items():
            extras[key].append(arr[start:h_end])

    n = len(starts_list)
    x_arr = np.stack(xs).astype(np.float32) if n else np.zeros((0, history_len, n_channels), np.float32)
    t_arr = np.stack(ts).astype(np.float32) if n else np.zeros((0, history_len), np.float32)
    y_arr = np.stack(ys).astype(np.float32) if n else np.zeros((0, pred_len, n_channels), np.float32)
    mask_arr = np.ones_like(x_arr, dtype=np.float32)
    starts_arr = np.asarray(starts_list, dtype=np.int64) + row_offset
    ends_arr = starts_arr + span

    condition = None
    if extra_channels and "condition" in extras and n:
        condition = np.stack(extras["condition"]).astype(np.float32)

    return SplitArrays(
        name=split_name,
        x=x_arr,
        t=t_arr,
        y=y_arr,
        y_raw=y_arr.copy(),
        mask=mask_arr,
        starts=starts_arr,
        ends=ends_arr,
        condition=condition,
        meta={"row_offset": row_offset, "n_rows": total, "stride": stride,
              "history_len": history_len, "pred_len": pred_len},
    )


def hash_identity(payload: Dict) -> str:
    """把 split/协议描述序列化后取 sha256 前 16 位作为身份指纹。"""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
