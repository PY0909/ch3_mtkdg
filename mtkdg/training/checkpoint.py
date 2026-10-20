"""checkpoint 保存与恢复（best / last）。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

CHECKPOINT_FORMAT = "ch3_checkpoint_v1"


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unexpected checkpoint format in {path}")
    if model is not None:
        model.load_state_dict(payload["state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
