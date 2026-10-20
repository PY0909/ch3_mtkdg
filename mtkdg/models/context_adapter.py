"""Train-fitted context metadata and a support-shrunk residual adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ContextAssignment:
    difficulty_bin: np.ndarray
    support_weight: np.ndarray
    group_keys: np.ndarray


class ContextMetadata:
    """Train-only OOF difficulty boundaries and independent-cluster supports."""

    def __init__(self, feature_mean, feature_scale, ridge_coef, ridge_intercept,
                 bin_edges, support_counts, kappa_train, fit_split="train"):
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.ridge_coef = np.asarray(ridge_coef, dtype=np.float64)
        self.ridge_intercept = float(ridge_intercept)
        self.bin_edges = np.asarray(bin_edges, dtype=np.float64)
        self.support_counts = {int(k): int(v) for k, v in support_counts.items()}
        self.kappa_train = float(kappa_train)
        self.fit_split = fit_split

    @classmethod
    def fit_train(cls, features, point_error, cluster_id, condition_code=None,
                  split="train", ridge_alpha=1.0, n_folds=5):
        if split != "train":
            raise ValueError("ContextMetadata may only be fit on split='train'")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(point_error, dtype=np.float64).reshape(-1)
        clusters = np.asarray(cluster_id).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(clusters):
            raise ValueError("features, point_error, and cluster_id must align")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("train context inputs must be finite")
        codes = np.zeros(len(y), dtype=np.int64) if condition_code is None else np.asarray(condition_code).reshape(-1)
        if len(codes) != len(y):
            raise ValueError("condition_code must align with features")
        unique = np.unique(clusters)
        if len(unique) < 2:
            raise ValueError("at least two independent clusters are required")
        n_splits = min(int(n_folds), len(unique))
        folds = np.array_split(unique, n_splits)
        oof = np.empty(len(y), dtype=np.float64)
        for fold in folds:
            val_mask = np.isin(clusters, fold)
            train_mask = ~val_mask
            mean = x[train_mask].mean(axis=0)
            scale = x[train_mask].std(axis=0)
            scale[scale < 1e-8] = 1.0
            xt = (x[train_mask] - mean) / scale
            xv = (x[val_mask] - mean) / scale
            coef, intercept = _ridge_fit(xt, y[train_mask], ridge_alpha)
            oof[val_mask] = xv @ coef + intercept
        edges = np.quantile(oof, [0.25, 0.5, 0.75])
        group_keys = _combine_context(codes, np.searchsorted(edges, oof, side="right"))
        support_counts = {}
        for key in np.unique(group_keys):
            support_counts[int(key)] = int(len(np.unique(clusters[group_keys == key])))
        positive = np.asarray([n for n in support_counts.values() if n > 0], dtype=np.float64)
        kappa = max(1.0, float(np.median(positive))) if len(positive) else 1.0
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        coef, intercept = _ridge_fit((x - mean) / scale, y, ridge_alpha)
        return cls(mean, scale, coef, intercept, edges, support_counts, kappa)

    def assign(self, features, condition_code=None) -> ContextAssignment:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_mean.size:
            raise ValueError("features have incompatible shape")
        difficulty = ((x - self.feature_mean) / self.feature_scale) @ self.ridge_coef + self.ridge_intercept
        bins = np.searchsorted(self.bin_edges, difficulty, side="right").astype(np.int64)
        codes = np.zeros(len(x), dtype=np.int64) if condition_code is None else np.asarray(condition_code).reshape(-1)
        if len(codes) != len(x):
            raise ValueError("condition_code must align with features")
        keys = _combine_context(codes, bins)
        weights = np.asarray([
            count / (count + self.kappa_train) if (count := self.support_counts.get(int(key), 0)) > 0 else 0.0
            for key in keys
        ], dtype=np.float32)
        return ContextAssignment(bins, weights, keys)

    def state_dict(self) -> Dict:
        state = {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "ridge_coef": self.ridge_coef.tolist(),
            "ridge_intercept": self.ridge_intercept,
            "bin_edges": self.bin_edges.tolist(),
            "support_counts": {str(k): v for k, v in self.support_counts.items()},
            "kappa_train": self.kappa_train,
            "fit_split": self.fit_split,
        }
        state["sha256"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return state


def _combine_context(codes, bins):
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    bins = np.asarray(bins, dtype=np.int64).reshape(-1)
    return codes * 4 + bins


def _ridge_fit(x, y, alpha):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    coef = np.linalg.solve(xc.T @ xc + float(alpha) * np.eye(x.shape[1]), xc.T @ yc)
    intercept = float(y_mean - x_mean @ coef)
    return coef, intercept


class ContextFeatureBuilder:
    """Flatten only inference-time encoder summaries into one context vector."""

    def __init__(self, d_model: int, n_channels: int, condition_dim: int = 0, n_scales: int = 3):
        self.d_model = int(d_model)
        self.n_channels = int(n_channels)
        self.condition_dim = int(condition_dim)
        self.n_scales = int(n_scales)
        self.feature_dim = 3 * self.d_model + self.n_scales + self.n_channels + 2 + self.condition_dim

    def __call__(self, encoder_output: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        z = encoder_output["z"]
        parts = [z.mean(dim=1), z.max(dim=1).values, z.std(dim=1, unbiased=False)]
        parts.extend((encoder_output["gate_weights"], encoder_output["obs_ratio"]))
        if encoder_output["gate_weights"].size(-1) != self.n_scales:
            raise ValueError("scale gate has incompatible width")
        alpha = encoder_output.get("fusion_alpha")
        parts.append(alpha.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
                     if alpha is not None else z.new_zeros((z.size(0), 1)))
        # A second stable summary keeps the feature contract fixed across fusion modes.
        parts.append(alpha.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
                     if alpha is not None else z.new_zeros((z.size(0), 1)))
        cond = encoder_output.get("cond_emb")
        if self.condition_dim:
            if cond is None:
                cond = z.new_zeros((z.size(0), self.condition_dim))
            if cond.size(-1) != self.condition_dim:
                raise ValueError("condition embedding has incompatible width")
            parts.append(cond)
        out = torch.cat(parts, dim=-1)
        if out.size(-1) != self.feature_dim:
            raise RuntimeError(f"context feature width mismatch: {out.size(-1)} != {self.feature_dim}")
        return out


@dataclass
class AdapterOutput:
    z_ctx: torch.Tensor
    delta: torch.Tensor
    gate: torch.Tensor
    support_weight: torch.Tensor


class ContextualResidualAdapter(nn.Module):
    def __init__(self, d_model: int, feature_dim: int, n_channels: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.n_channels = int(n_channels)
        hidden = int(hidden_dim or d_model)
        self.channel_embedding = nn.Embedding(n_channels, d_model)
        input_dim = 2 * d_model + feature_dim + d_model
        self.norm = nn.LayerNorm(input_dim)
        self.hidden = nn.Linear(input_dim, hidden)
        self.delta = nn.Linear(hidden, d_model)
        self.gate = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, encoder_output, context_features=None, support_weight=None,
                disable_adapter=False, permutation=None):
        z = encoder_output["z"] if isinstance(encoder_output, dict) else encoder_output
        if z.ndim != 3 or z.size(1) != self.n_channels or z.size(2) != self.d_model:
            raise ValueError("z must have shape [B, n_channels, d_model]")
        if context_features is None:
            raise ValueError("context_features are required")
        if permutation is not None:
            context_features = context_features[permutation]
        if context_features.shape[0] != z.shape[0]:
            raise ValueError("context feature batch size mismatch")
        if support_weight is None:
            support_weight = z.new_ones((z.size(0),))
        support_weight = torch.as_tensor(support_weight, dtype=z.dtype, device=z.device).reshape(-1)
        if len(support_weight) != z.size(0):
            raise ValueError("support_weight must have one value per sample")
        channel = self.channel_embedding.weight.unsqueeze(0).expand(z.size(0), -1, -1)
        global_mean = z.mean(dim=1, keepdim=True).expand_as(z)
        ctx = context_features.unsqueeze(1).expand(-1, self.n_channels, -1)
        h = torch.nn.functional.gelu(self.hidden(self.norm(torch.cat([z, global_mean, ctx, channel], dim=-1))))
        delta = self.delta(h)
        gate = torch.sigmoid(self.gate(h))
        weight = support_weight.clamp(0.0, 1.0).view(-1, 1, 1)
        if disable_adapter:
            weight = torch.zeros_like(weight)
        z_ctx = z + weight * gate * delta
        return AdapterOutput(z_ctx, delta, gate, weight)
