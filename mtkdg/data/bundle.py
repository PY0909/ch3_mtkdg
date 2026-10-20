"""统一数据 bundle：把 MetroPT / C-MAPSS 协议包装成同一访问接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..config import ExperimentConfig
from .cmapss import build_cmapss_splits
from .common import SplitArrays
from .metropt import build_metropt_splits, load_metropt_frame, metropt_csv_path
from .normalizer import TrainOnlyStandardScaler


@dataclass
class DatasetBundle:
    task: str                       # state_forecast | rul_regression
    dataset_name: str
    split_identity: str
    identity: str
    n_channels: int
    condition_dim: int              # 3 (FD002/FD004) | 0 (MetroPT)
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    feature_scaler: TrainOnlyStandardScaler
    target_scaler: TrainOnlyStandardScaler
    feature_schema: Dict
    target_schema: Dict
    meta: Dict = field(default_factory=dict)
    rul_mean: float = 0.0           # 仅 RUL 任务
    rul_std: float = 1.0
    rul_raw_mean: float = 0.0       # raw（未截断）标签统计（V2 消融用）
    rul_raw_std: float = 1.0

    def split(self, name: str) -> SplitArrays:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def sample_counts(self) -> Dict[str, int]:
        return {
            "n_train_samples": self.train.n_samples,
            "n_val_samples": self.val.n_samples,
            "n_test_samples": self.test.n_samples,
        }


def bundle_from_metropt_splits(s) -> DatasetBundle:
    return DatasetBundle(
        task="state_forecast",
        dataset_name="metropt3",
        split_identity=s.meta["split_identity"],
        identity=s.identity,
        n_channels=len(s.feature_schema["channels"]),
        condition_dim=0,
        train=s.train,
        val=s.val,
        test=s.test,
        feature_scaler=s.scaler,
        target_scaler=s.scaler,
        feature_schema=s.feature_schema,
        target_schema=s.target_schema,
        meta=s.meta | {"boundaries": s.boundaries},
    )


def bundle_from_cmapss_splits(s) -> DatasetBundle:
    raw = s.train.y_raw_uncapped
    raw_mean = float(raw.mean()) if raw is not None and len(raw) else 0.0
    raw_std = float(raw.std()) if raw is not None and len(raw) else 1.0
    return DatasetBundle(
        task="rul_regression",
        dataset_name=s.subset.lower(),   # fd002 / fd004
        split_identity=s.split_identity,
        identity=s.identity,
        n_channels=len(s.feature_schema["channels"]),
        condition_dim=3,
        train=s.train,
        val=s.val,
        test=s.test,
        feature_scaler=s.feature_scaler,
        target_scaler=s.target_scaler,
        feature_schema=s.feature_schema,
        target_schema=s.target_schema,
        meta=dict(s.meta),
        rul_mean=float(s.target_scaler.mean[0]),
        rul_std=float(s.target_scaler.std[0]),
        rul_raw_mean=raw_mean,
        rul_raw_std=raw_std,
    )


def _exclude_degenerate_channels(bundle: DatasetBundle, threshold: float = 0.05) -> DatasetBundle:
    """V3 消融：把 train std < threshold（含 min_std 下限后的值）的通道从 encoder
    输入剔除（x 置 0、mask 置 0），输出头与目标不变（[B,P,C] 合同保持）。"""
    stds = np.asarray(bundle.feature_scaler.std, dtype=np.float64)
    names = bundle.feature_schema["channels"]
    drop = [i for i, s in enumerate(stds) if s < threshold]
    if not drop:
        return bundle
    for split in (bundle.train, bundle.val, bundle.test):
        split.x[:, :, drop] = 0.0
        split.mask[:, :, drop] = 0.0
    bundle.meta["excluded_degenerate_channels"] = [names[i] for i in drop]
    return bundle


def build_bundle(cfg: ExperimentConfig, nrows: int = 0) -> DatasetBundle:
    """按 dataset 构造统一 bundle。`nrows>0` 仅用于 smoke（截断 MetroPT 读取）。"""
    if cfg.dataset == "metropt3":
        frame = load_metropt_frame(metropt_csv_path(cfg.data_root), nrows=nrows or None)
        s = build_metropt_splits(
            frame,
            history_len=cfg.history_len,
            pred_len=cfg.pred_len,
            stride=cfg.stride,
        )
        bundle = bundle_from_metropt_splits(s)
        if cfg.metropt_exclude_degenerate:
            bundle = _exclude_degenerate_channels(bundle)
        return bundle

    s = build_cmapss_splits(
        data_root=cfg.data_root,
        subset=cfg.dataset.upper(),
        history_len=cfg.history_len,
        stride=cfg.stride,
        training_cap=cfg.training_cap,
        training_target=cfg.training_target,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
        norm_mode=cfg.norm_mode,
    )
    return bundle_from_cmapss_splits(s)
