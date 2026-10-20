"""C-MAPSS FD002/FD004 数据 adapter（engine-level 协议）。

协议（固定）：
1. 读取官方 train/test/RUL 三件套，按 (engine_id, cycle) 排序；
2. 删除近似常数传感器（名单固定在配置中，FD002/FD004 分别记录），
   特征 = 3 个 operating settings + 14 个有效传感器 = 17 通道；
3. 训练 engine 按 engine_id 划分 train/validation（先切 engine 再切窗口，
   禁止同一 engine 的窗口跨 split）；
4. 官方 test engine 只用于最终 test，每 engine 只保留末端窗口；
   历史不足 H 时首观测左填充并记录 history_padding；
5. train/validation 标签使用 capped RUL（默认 125），test 使用官方 raw RUL；
6. 归一化统计量只由 train engines 拟合；
7. FD002 与 FD004 完全独立，分开训练、分开报告。
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import SplitArrays, hash_identity
from .normalizer import TrainOnlyStandardScaler

SETTING_COLUMNS = ["setting1", "setting2", "setting3"]

# 近似常数通道名单（配置固定；FC-STGNN 同款名单，FD002/FD004 一致）
DROPPED_SENSORS_FD002 = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
DROPPED_SENSORS_FD004 = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

ALL_SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]

CMAPSS_COLUMNS = ["id", "cycle"] + SETTING_COLUMNS + ALL_SENSOR_COLUMNS

VALID_SUBSETS = ("FD002", "FD004")


def retained_channels(subset: str) -> List[str]:
    dropped = DROPPED_SENSORS_FD002 if subset == "FD002" else DROPPED_SENSORS_FD004
    sensors = [c for c in ALL_SENSOR_COLUMNS if c not in dropped]
    return SETTING_COLUMNS + sensors


def split_identity_for(subset: str) -> str:
    return f"cmapss_{subset.lower()}_engine_level_v1"


def _read_cmapss_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"{path}: expected {len(CMAPSS_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = CMAPSS_COLUMNS
    return df.sort_values(["id", "cycle"], kind="stable").reset_index(drop=True)


def load_cmapss_train(data_root: str, subset: str) -> pd.DataFrame:
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset}")
    path = os.path.join(data_root, "CMAPSSData", f"train_{subset}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _read_cmapss_txt(path)


def load_cmapss_test(data_root: str, subset: str) -> Tuple[pd.DataFrame, np.ndarray]:
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset}")
    path = os.path.join(data_root, "CMAPSSData", f"test_{subset}.txt")
    truth_path = os.path.join(data_root, "CMAPSSData", f"RUL_{subset}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not os.path.exists(truth_path):
        raise FileNotFoundError(truth_path)
    df = _read_cmapss_txt(path)
    truth = pd.read_csv(truth_path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    truth = truth.iloc[:, 0].to_numpy(dtype=np.float64)
    return df, truth


def engine_level_split(
    engine_ids: List[int],
    val_ratio: float = 0.2,
    seed: int = 2026,
) -> Tuple[List[int], List[int]]:
    """按 engine 划分 train/validation（engine 为最小单位，deterministic）。

    返回 (train_engines, val_engines)，均为升序列表。
    """
    ids = np.array(sorted(engine_ids), dtype=np.int64)
    rng = np.random.RandomState(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_engines = sorted(int(e) for e in shuffled[:n_val])
    train_engines = sorted(int(e) for e in shuffled[n_val:])
    if not train_engines:
        raise ValueError("engine split produced empty train set")
    return train_engines, val_engines


def _windows_for_engine(
    feats: np.ndarray,          # [L, 17] 已标准化
    cycles: np.ndarray,         # [L] cycle 序号
    history_len: int,
    stride: int,
    labels_row: np.ndarray,     # [L] 每行 raw RUL（train 引擎专用）
    last_window_only: bool,
    fixed_label: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float], List[int]]:
    """切一个 engine 的窗口。

    - last_window_only=False：滑窗，标签 = 窗口末端 RUL（可 capped）；
    - last_window_only=True：只取末端窗口，标签 = fixed_label（官方 raw RUL）；
      长度不足 H 时用首观测左填充。
    返回 (xs, ts, conds, labels, n_pads)
    """
    L = feats.shape[0]
    xs, ts, conds, labels, pads = [], [], [], [], []
    if last_window_only:
        if L >= history_len:
            window = feats[L - history_len:L]
            t_win = cycles[L - history_len:L]
            pad = 0
        else:
            pad = history_len - L
            window = np.concatenate(
                [np.repeat(feats[:1], pad, axis=0), feats], axis=0
            )
            t_win = np.concatenate(
                [np.repeat(cycles[:1], pad), cycles]
            )
        xs.append(window.astype(np.float32))
        ts.append(t_win.astype(np.float32))
        conds.append(window[:, :3].astype(np.float32))
        labels.append(float(fixed_label))
        pads.append(int(pad))
        return xs, ts, conds, labels, pads

    if L < history_len:
        return xs, ts, conds, labels, pads
    for start in range(0, L - history_len + 1, stride):
        end = start + history_len
        xs.append(feats[start:end].astype(np.float32))
        ts.append(cycles[start:end].astype(np.float32))
        conds.append(feats[start:end, :3].astype(np.float32))
        labels.append(float(labels_row[end - 1]))
        pads.append(0)
    return xs, ts, conds, labels, pads


def condition_codes(settings: np.ndarray) -> np.ndarray:
    """工况编码（V1 冻结规则）：三工况各保留 2 位小数组成元组后 factorize。

    train/test 行统一编码（编码仅由 settings 值决定，不含任何标签信息）。
    """
    rounded = np.round(np.asarray(settings, dtype=np.float64), 2)
    tuples = [tuple(row) for row in rounded]
    uniq = sorted(set(tuples))
    index = {t: i for i, t in enumerate(uniq)}
    return np.asarray([index[t] for t in tuples], dtype=np.int64)


@dataclass
class CMAPSSSplits:
    subset: str
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    feature_scaler: TrainOnlyStandardScaler
    target_scaler: TrainOnlyStandardScaler
    identity: str
    split_identity: str
    train_engines: List[int]
    val_engines: List[int]
    meta: Dict = field(default_factory=dict)

    @property
    def feature_schema(self) -> Dict:
        channels = retained_channels(self.subset)
        return {
            "channels": channels,
            "n_channels": len(channels),
            "dropped_sensors": (
                DROPPED_SENSORS_FD002 if self.subset == "FD002" else DROPPED_SENSORS_FD004
            ),
            "shape": "[N, H, C]",
        }

    @property
    def target_schema(self) -> Dict:
        return {
            "shape": "[N, 1]",
            "unit": "raw cycle",
            "training_target": self.meta.get("training_target", "capped"),
            "training_cap": self.meta.get("training_cap"),
            "train_val": (
                "complete raw RUL" if self.meta.get("training_target") == "raw"
                else "capped at training_cap"
            ),
            "test": "official raw RUL (uncapped)",
        }


def _label_for_training(raw_rul: np.ndarray, training_target: str, training_cap: int) -> np.ndarray:
    """Return the explicitly selected train/validation RUL target protocol."""
    values = np.asarray(raw_rul, dtype=np.float64)
    if training_target == "raw":
        return values.copy()
    if training_target != "capped":
        raise ValueError("training_target must be 'raw' or 'capped'")
    if not isinstance(training_cap, (int, np.integer)) or training_cap <= 0:
        raise ValueError("training_cap must be a positive integer for capped targets")
    return np.minimum(values, float(training_cap))


def build_cmapss_splits(
    data_root: str,
    subset: str,
    history_len: int = 50,
    stride: int = 1,
    training_cap: int = 125,
    training_target: str = "capped",
    val_ratio: float = 0.2,
    split_seed: int = 2026,
    norm_mode: str = "global",
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_truth: Optional[np.ndarray] = None,
) -> CMAPSSSplits:
    """构造 FD002/FD004 的 train/val/test 窗口数组。

    `train_df`/`test_df`/`test_truth` 可注入合成数据（测试用）；缺省读取官方文件。
    """
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset}")
    if training_target not in ("raw", "capped"):
        raise ValueError("training_target must be 'raw' or 'capped'")
    if training_target == "capped" and (
        not isinstance(training_cap, (int, np.integer)) or training_cap <= 0
    ):
        raise ValueError("training_cap must be a positive integer for capped targets")
    if train_df is None:
        train_df = load_cmapss_train(data_root, subset)
    if test_df is None or test_truth is None:
        test_df, test_truth = load_cmapss_test(data_root, subset)

    channels = retained_channels(subset)
    dropped = DROPPED_SENSORS_FD002 if subset == "FD002" else DROPPED_SENSORS_FD004

    # 1) engine-level split（先切 engine，再切窗口）
    all_engines = sorted(int(e) for e in train_df["id"].unique())
    train_engines, val_engines = engine_level_split(all_engines, val_ratio, split_seed)

    # 2) raw RUL per train row：max_cycle - cycle
    max_cycle = train_df.groupby("id")["cycle"].max().rename("max_cycle")
    train_df = train_df.merge(max_cycle, on="id", how="left")
    train_df["rul_raw"] = train_df["max_cycle"] - train_df["cycle"]

    feats_all = train_df[channels].to_numpy(dtype=np.float64)
    train_mask = train_df["id"].isin(train_engines).to_numpy()

    # 3) 归一化只由 train engines 拟合
    #    - global（主协议）：单一 z-score
    #    - per_condition（V1 消融）：6 工况各拟 scaler（min_std=0.01），未见工况回退全局
    from .normalizer import PerConditionScaler

    if norm_mode == "per_condition":
        codes_all = condition_codes(train_df[SETTING_COLUMNS].to_numpy(dtype=np.float64))
        feature_scaler = PerConditionScaler(min_std=0.01)
        feature_scaler.fit(feats_all[train_mask], codes_all[train_mask])
        feats_norm = feature_scaler.transform(feats_all, codes_all)
    elif norm_mode == "global":
        feature_scaler = TrainOnlyStandardScaler()
        feature_scaler.fit(feats_all[train_mask])
        feats_norm = feature_scaler.transform(feats_all)
    else:
        raise ValueError(f"unknown norm_mode: {norm_mode}")

    # 被删通道在 train engines 上的 std（审计记录，证明近常数）
    dropped_audit = {
        c: float(train_df.loc[train_mask, c].std()) for c in dropped
    }

    cycles_all = train_df["cycle"].to_numpy(dtype=np.float64)
    rul_all = train_df["rul_raw"].to_numpy(dtype=np.float64)
    ids_all = train_df["id"].to_numpy(dtype=np.int64)
    id_to_pos = {int(e): i for i, e in enumerate(all_engines)}

    def _build_split(engines: List[int], split_name: str) -> SplitArrays:
        xs, ts, conds, labels, pads, eng_of_sample = [], [], [], [], [], []
        labels_uncapped = []
        for eng in engines:
            rows = np.flatnonzero(ids_all == eng)
            _xs, _ts, _conds, _labels, _pads = _windows_for_engine(
                feats_norm[rows],
                cycles_all[rows],
                history_len,
                stride,
                labels_row=rul_all[rows],
                last_window_only=False,
            )
            # Historical capped mode remains the default; raw mode is explicit.
            for i, _ in enumerate(_xs):
                lab = float(_label_for_training(_labels[i], training_target, training_cap))
                xs.append(_xs[i]); ts.append(_ts[i]); conds.append(_conds[i])
                labels.append(lab); pads.append(_pads[i])
                labels_uncapped.append(_labels[i])
                eng_of_sample.append(eng)
        if not xs:
            raise RuntimeError(f"split for engines {engines[:5]}... produced 0 windows")
        x = np.stack(xs); t = np.stack(ts); cond = np.stack(conds)
        y_raw = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
        mask = np.ones_like(x, dtype=np.float32)
        return SplitArrays(
            name=split_name,
            x=x.astype(np.float32),
            t=t.astype(np.float32),
            y=y_raw.astype(np.float32),   # 占位，随后由 target_scaler 标准化
            y_raw=y_raw.astype(np.float32),
            mask=mask,
            starts=np.arange(len(xs), dtype=np.int64),
            ends=np.arange(len(xs), dtype=np.int64),
            condition=cond.astype(np.float32),
            # 未截断标签仅作监控诊断（val_RMSE_rawtarget），协议上不可用于选择 checkpoint
            y_raw_uncapped=np.asarray(labels_uncapped, dtype=np.float64).reshape(-1, 1),
            meta={
                "engine_ids": np.asarray(eng_of_sample, dtype=np.int64),
                "n_padded": int(sum(1 for p in pads if p > 0)),
                "padding_amounts": [p for p in pads if p > 0],
            },
        )

    train_split = _build_split(train_engines, "train")
    val_split = _build_split(val_engines, "val")

    # 4) target scaler is fitted only on train-engine labels under this protocol.
    target_scaler = TrainOnlyStandardScaler()
    target_scaler.fit(train_split.y_raw.astype(np.float64))
    for s in (train_split, val_split):
        s.y = target_scaler.transform(s.y_raw)

    # 5) 官方 test：每 engine 只保留末端窗口，标签 = 官方 raw RUL（不 cap）
    test_feats_all = test_df[channels].to_numpy(dtype=np.float64)
    if norm_mode == "per_condition":
        test_codes = condition_codes(test_df[SETTING_COLUMNS].to_numpy(dtype=np.float64))
        test_feats_norm = feature_scaler.transform(test_feats_all, test_codes)
    else:
        test_feats_norm = feature_scaler.transform(test_feats_all)
    test_ids = test_df["id"].to_numpy(dtype=np.int64)
    test_cycles = test_df["cycle"].to_numpy(dtype=np.float64)
    xs, ts, conds, labels, pads, eng_of_sample = [], [], [], [], [], []
    for i, eng in enumerate(sorted(int(e) for e in np.unique(test_ids))):
        if i >= len(test_truth):
            break
        rows = np.flatnonzero(test_ids == eng)
        _xs, _ts, _conds, _labels, _pads = _windows_for_engine(
            test_feats_norm[rows],
            test_cycles[rows],
            history_len,
            stride,
            labels_row=None,
            last_window_only=True,
            fixed_label=float(test_truth[i]),
        )
        for j in range(len(_xs)):
            xs.append(_xs[j]); ts.append(_ts[j]); conds.append(_conds[j])
            labels.append(_labels[j]); pads.append(_pads[j])
            eng_of_sample.append(eng)
    x = np.stack(xs); t = np.stack(ts); cond = np.stack(conds)
    y_raw = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    test_split = SplitArrays(
        name="test",
        x=x.astype(np.float32),
        t=t.astype(np.float32),
        y=target_scaler.transform(y_raw),
        y_raw=y_raw.astype(np.float32),
        mask=np.ones_like(x, dtype=np.float32),
        starts=np.arange(len(xs), dtype=np.int64),
        ends=np.arange(len(xs), dtype=np.int64),
        condition=cond.astype(np.float32),
        meta={
            "engine_ids": np.asarray(eng_of_sample, dtype=np.int64),
            "n_padded": int(sum(1 for p in pads if p > 0)),
            "padding_amounts": [p for p in pads if p > 0],
        },
    )

    split_identity = split_identity_for(subset)
    meta = {
        "subset": subset,
        "history_len": history_len,
        "stride": stride,
        "training_cap": training_cap,
        "training_target": training_target,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        "norm_mode": norm_mode,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "n_test_engines": int(len(np.unique(test_ids))),
        "dropped_sensor_std_train": dropped_audit,
        "test_label_source": "official RUL file (raw, uncapped)",
        "train_label_sha256": hashlib.sha256(train_split.y_raw.tobytes()).hexdigest(),
        "val_label_sha256": hashlib.sha256(val_split.y_raw.tobytes()).hexdigest(),
        "target_scaler_identity": target_scaler.identity(),
    }
    identity = hash_identity({
        "protocol": split_identity,
        "subset": subset,
        "train_engines": train_engines,
        "val_engines": val_engines,
        "history_len": history_len,
        "stride": stride,
        "training_cap": training_cap,
        "training_target": training_target,
        "feature_scaler": feature_scaler.identity(),
        "target_scaler": target_scaler.identity(),
    })

    return CMAPSSSplits(
        subset=subset,
        train=train_split,
        val=val_split,
        test=test_split,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        identity=identity,
        split_identity=split_identity,
        train_engines=train_engines,
        val_engines=val_engines,
        meta=meta,
    )
