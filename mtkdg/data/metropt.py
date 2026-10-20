"""MetroPT-3 AirCompressor 数据 adapter（首月协议）。

协议（固定，不可随机切分）：
1. 按 `timestamp` 稳定升序排序，丢弃行号列，保留 15 个数值传感器；
2. 第一自然月：前 80% *时间* 为 train，后 20% 为 validation；
3. 第二个自然月及以后全部为 test；
4. 窗口 [i, i+H+P) 必须完整落在同一 split 内，禁止跨边界；
5. 归一化统计量只由 train 时间段拟合，应用到 val/test；
6. test 只能使用 validation-best checkpoint 评估一次。

split identity: `metropt3_official_first_month_v1`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import SplitArrays, build_contiguous_windows, hash_identity
from .normalizer import TrainOnlyStandardScaler

METROPT_SENSOR_COLUMNS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current", "COMP", "DV_eletric",
    "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
    "Caudal_impulses",
]

SPLIT_IDENTITY = "metropt3_official_first_month_v1"
TRAIN_FRACTION = 0.8
# pre-registered（Round 1 review W3）：train 内近常数通道（Oil_level std=0，
# MPG/LPS 近退化）的 std 下限，防止 test 真实波动被放大为 10^3 sigma 输入。
METROPT_MIN_STD = 0.01


def load_metropt_frame(csv_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """读取原始 CSV，排序并构造时间坐标。"""
    df = pd.read_csv(csv_path, nrows=nrows)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=unnamed)
    missing = [c for c in METROPT_SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing sensor columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    t0 = df["timestamp"].iloc[0]
    df["t_rel"] = (df["timestamp"] - t0).dt.total_seconds()
    df["delta_t"] = df["t_rel"].diff()
    df.loc[df.index[0], "delta_t"] = 0.0
    return df


def first_month_boundaries(timestamps: pd.Series) -> Dict:
    """返回首月协议的时间边界。

    - month_start：首行所在自然月的第一天 00:00
    - month_end：下一自然月第一天 00:00
    - train_val_ts：month_start + 80% * 月时长（按时间切，不按行数切）
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    month_start = ts[0].normalize()
    month_end = month_start + pd.offsets.MonthBegin(1)
    month_seconds = (month_end - month_start).total_seconds()
    train_val_ts = month_start + pd.Timedelta(
        seconds=TRAIN_FRACTION * month_seconds
    )
    return {
        "month_start": month_start,
        "train_val_ts": train_val_ts,
        "month_end": month_end,
    }


def assign_split_labels(df: pd.DataFrame) -> np.ndarray:
    """按时间戳给每行打 split 标签（'train'/'val'/'test'）。"""
    b = first_month_boundaries(df["timestamp"])
    ts = pd.DatetimeIndex(df["timestamp"])
    labels = np.full(len(df), "test", dtype=object)
    in_first_month = ts < b["month_end"]
    is_train = ts < b["train_val_ts"]
    labels[in_first_month & is_train] = "train"
    labels[in_first_month & ~is_train] = "val"
    labels[~in_first_month] = "test"
    return labels.astype(str)


def gap_statistics(delta_t: pd.Series, threshold_s: float = 60.0) -> Dict:
    """超长断档审计：保留 gap 特征但记录统计，不默默删除。"""
    delta = delta_t.to_numpy(dtype=np.float64)
    finite = delta[np.isfinite(delta)]
    return {
        "max_delta_s": float(finite.max()) if len(finite) else 0.0,
        "median_delta_s": float(np.median(finite)) if len(finite) else 0.0,
        "n_gaps_gt_60s": int((finite > threshold_s).sum()),
        "n_gaps_gt_600s": int((finite > 600.0).sum()),
    }


@dataclass
class MetroPTSplits:
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    scaler: TrainOnlyStandardScaler          # 特征与目标共用（同一组传感器）
    identity: str
    boundaries: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    @property
    def feature_schema(self) -> Dict:
        return {
            "channels": list(METROPT_SENSOR_COLUMNS),
            "n_channels": len(METROPT_SENSOR_COLUMNS),
            "shape": "[N, H, C]",
        }

    @property
    def target_schema(self) -> Dict:
        return {
            "channels": list(METROPT_SENSOR_COLUMNS),
            "shape": "[N, P, C]",
            "unit": "raw sensor unit (metrics computed after inverse transform)",
        }


def _ensure_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """保证 frame 含 t_rel / delta_t（load_metropt_frame 已加；合成数据兜底）。"""
    frame = frame.copy()
    if "t_rel" not in frame.columns:
        ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
        frame["t_rel"] = (ts - ts[0]).total_seconds()
    if "delta_t" not in frame.columns:
        frame["delta_t"] = frame["t_rel"].diff()
        frame.loc[frame.index[0], "delta_t"] = 0.0
    return frame


def build_metropt_splits(
    frame: pd.DataFrame,
    history_len: int = 168,
    pred_len: int = 24,
    stride: int = 60,
) -> MetroPTSplits:
    """从已排序的 frame 构造三个 split 的窗口数组。"""
    frame = _ensure_time_columns(frame)
    labels = assign_split_labels(frame)
    raw = frame[METROPT_SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    times = frame["t_rel"].to_numpy(dtype=np.float32)

    # 1) 先在原始单位上切窗（窗口不跨 split 边界由行范围保证）
    row_ranges: Dict[str, tuple] = {}
    for split in ("train", "val", "test"):
        idx = np.flatnonzero(labels == split)
        if len(idx) == 0:
            raise ValueError(f"MetroPT split '{split}' is empty; check first-month boundaries")
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        # 协议要求每个 split 的行是连续区间；若数据本身乱序导致交错则报错
        if hi - lo != len(idx):
            raise ValueError(f"split '{split}' rows are not contiguous; sort by timestamp first")
        row_ranges[split] = (lo, hi)

    built = {}
    for split, (lo, hi) in row_ranges.items():
        built[split] = build_contiguous_windows(
            values=raw[lo:hi],
            times=times[lo:hi],
            history_len=history_len,
            pred_len=pred_len,
            stride=stride,
            split_name=split,
            row_offset=lo,
        )

    # 2) normalizer 只由 train 拟合（原始单位），再应用到全部 split
    #    min_std=0.01（pre-registered，见 METROPT_MIN_STD 注释）
    train_lo, train_hi = row_ranges["train"]
    scaler = TrainOnlyStandardScaler(min_std=METROPT_MIN_STD)
    scaler.fit(raw[train_lo:train_hi])
    for split in ("train", "val", "test"):
        arr = built[split]
        arr.x = scaler.transform(arr.x)
        arr.y = scaler.transform(arr.y)
        # y_raw 保留原始单位，评估指标在 y_raw 上计算
        # W4 数据基础：每个窗口起始行的自然月标签（供逐月 test 分解）
        months = frame["timestamp"].dt.strftime("%Y-%m").to_numpy()
        arr.meta["window_months"] = months[arr.starts].tolist()
        target_starts = arr.starts + history_len
        target_ends = arr.starts + history_len + pred_len
        arr.meta["origin_timestamps"] = frame["timestamp"].iloc[arr.starts].astype(str).tolist()
        arr.meta["target_start_timestamps"] = frame["timestamp"].iloc[target_starts].astype(str).tolist()
        arr.meta["target_end_timestamps"] = frame["timestamp"].iloc[target_ends - 1].astype(str).tolist()

    b = first_month_boundaries(frame["timestamp"])
    boundaries = {
        "month_start": str(b["month_start"]),
        "train_val_ts": str(b["train_val_ts"]),
        "month_end": str(b["month_end"]),
        "train_rows": [row_ranges["train"][0], row_ranges["train"][1]],
        "val_rows": [row_ranges["val"][0], row_ranges["val"][1]],
        "test_rows": [row_ranges["test"][0], row_ranges["test"][1]],
    }
    meta = {
        "split_identity": SPLIT_IDENTITY,
        "history_len": history_len,
        "pred_len": pred_len,
        "stride": stride,
        "normalizer_min_std": METROPT_MIN_STD,
        "n_rows_total": int(len(frame)),
        "gap_stats_train": gap_statistics(frame["delta_t"].iloc[row_ranges["train"][0]:row_ranges["train"][1]]),
        "gap_stats_all": gap_statistics(frame["delta_t"]),
    }
    identity = hash_identity({
        "protocol": SPLIT_IDENTITY,
        "boundaries": boundaries,
        "history_len": history_len,
        "pred_len": pred_len,
        "stride": stride,
        "scaler": scaler.identity(),
    })

    return MetroPTSplits(
        train=built["train"],
        val=built["val"],
        test=built["test"],
        scaler=scaler,
        identity=identity,
        boundaries=boundaries,
        meta=meta,
    )


def metropt_csv_path(data_root: str) -> str:
    path = os.path.join(data_root, "metropt+3+dataset", "MetroPT3(AirCompressor).csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MetroPT3 csv not found: {path}")
    return path
