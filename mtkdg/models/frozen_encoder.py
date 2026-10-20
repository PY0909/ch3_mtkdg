"""Audited loader for the Chapter 3 MTK-DG encoder."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import MTKDGConfig
from ..training.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from .mtkdg import MTKDGEncoder


def _fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FrozenEncoder(nn.Module):
    """Encoder wrapper whose ``train`` call cannot enable encoder dropout."""

    def __init__(self, encoder: MTKDGEncoder, metadata: Optional[Dict] = None):
        super().__init__()
        self.encoder = encoder
        self.metadata = metadata or {}
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, t, mask=None, condition=None):
        with torch.no_grad():
            return self.encoder(x, t, mask, condition)

    def fingerprint(self) -> str:
        return _fingerprint(self.encoder)


def _config_from_checkpoint(payload: Dict, dataset: str) -> MTKDGConfig:
    config = payload.get("extra", {}).get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing extra.config")
    checkpoint_dataset = config.get("dataset")
    if checkpoint_dataset != dataset:
        raise ValueError(f"dataset mismatch: checkpoint={checkpoint_dataset}, requested={dataset}")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint config is missing model")
    allowed = set(MTKDGConfig.__dataclass_fields__)
    unknown = set(model_config) - allowed
    if unknown:
        raise ValueError(f"unknown model config fields: {sorted(unknown)}")
    return MTKDGConfig(**{key: model_config[key] for key in allowed if key in model_config})


def load_frozen_encoder(path: str, dataset: str, map_location: str = "cpu") -> FrozenEncoder:
    payload = load_checkpoint(path, map_location=map_location)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unexpected checkpoint format")
    dataset = dataset.lower()
    if dataset not in ("fd002", "fd004", "metropt3"):
        raise ValueError(f"unsupported dataset: {dataset}")
    model_cfg = _config_from_checkpoint(payload, dataset)
    state = payload.get("state_dict", {})
    encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError("checkpoint has no encoder state")
    encoder = MTKDGEncoder(
        n_channels=15 if dataset == "metropt3" else 17,
        d_model=model_cfg.d_model,
        n_kernels=model_cfg.n_kernels,
        scales=model_cfg.scales,
        graph_mode=model_cfg.graph_mode,
        graph_layers=model_cfg.graph_layers,
        graph_self_loop=model_cfg.graph_self_loop,
        fusion_mode=model_cfg.fusion_mode,
        share_kernels_across_scales=model_cfg.share_kernels_across_scales,
        condition_dim=0 if dataset == "metropt3" else 3,
        dropout=model_cfg.dropout,
        availability_mode=model_cfg.availability_mode,
    )
    expected = set(encoder.state_dict())
    actual = set(encoder_state)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(f"encoder state mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    encoder.load_state_dict(encoder_state, strict=True)
    return FrozenEncoder(encoder, metadata={"dataset": dataset, "checkpoint": path})
