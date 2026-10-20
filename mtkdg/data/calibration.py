"""Distribution-matched validation views used only for probability calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .common import SplitArrays


def make_calibration_view(split: SplitArrays, task: str, pred_len: int = 1) -> SplitArrays:
    if split.name != "val":
        raise ValueError("calibration views can only be derived from validation")
    if task == "rul_regression":
        if "engine_ids" not in split.meta:
            raise ValueError("FD validation split is missing engine_ids")
        engines = np.asarray(split.meta["engine_ids"])
        selected = np.asarray([np.flatnonzero(engines == eng)[-1] for eng in np.unique(engines)])
        cluster_id = engines[selected]
    elif task == "state_forecast":
        if pred_len <= 0:
            raise ValueError("pred_len must be positive for MetroPT")
        if "target_start_timestamps" not in split.meta:
            raise ValueError("MetroPT validation split is missing target timestamps")
        order = np.argsort(split.starts, kind="stable")
        chosen, last_end = [], None
        for idx in order:
            start, end = int(split.starts[idx]), int(split.ends[idx])
            if last_end is None or start + split.x.shape[1] >= last_end:
                chosen.append(int(idx))
                last_end = end
        selected = np.asarray(chosen, dtype=np.int64)
        target_starts = pd.to_datetime(np.asarray(split.meta["target_start_timestamps"])[selected])
        cluster_id = target_starts.strftime("%G-%V").to_numpy()
    else:
        raise ValueError(f"unsupported task: {task}")
    meta = dict(split.meta)
    meta["calibration_view"] = True
    meta["cluster_id"] = np.asarray(cluster_id)
    return SplitArrays(
        name="val_calibration",
        x=split.x[selected], t=split.t[selected], y=split.y[selected],
        y_raw=split.y_raw[selected], mask=split.mask[selected],
        starts=split.starts[selected], ends=split.ends[selected],
        condition=None if split.condition is None else split.condition[selected],
        y_raw_uncapped=None if split.y_raw_uncapped is None else split.y_raw_uncapped[selected],
        meta=meta,
    )
