"""Validation-only cluster-weighted quantile calibration and scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from ..models.prob_heads import QUANTILE_LEVELS


@dataclass
class ShrunkQuantileCalibrator:
    quantile_levels: np.ndarray
    global_correction: np.ndarray
    local_correction: Dict[int, np.ndarray]
    effective_support: Dict[int, int]
    kappa_val: float
    min_local_clusters: int = 5
    target_unit: str = "unspecified"

    @classmethod
    def fit(cls, predictions, targets, context_key, cluster_id, split,
            quantile_levels=QUANTILE_LEVELS, min_local_clusters=5,
            sample_weights=None, target_unit="unspecified"):
        if split != "val_calibration":
            raise ValueError("calibration may only fit split='val_calibration'")
        q = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if q.ndim < 2 or q.shape[:-1] != y.shape:
            raise ValueError("predictions must have shape targets.shape + [Q]")
        levels = np.asarray(quantile_levels, dtype=np.float64)
        if q.shape[-1] != len(levels) or np.any(np.diff(levels) <= 0):
            raise ValueError("quantile levels must be strictly increasing and match predictions")
        n = q.shape[0]
        keys = np.asarray(context_key).reshape(-1)
        clusters = np.asarray(cluster_id).reshape(-1)
        if len(keys) != n or len(clusters) != n:
            raise ValueError("context_key and cluster_id must align with first prediction dimension")
        if not np.isfinite(q).all() or not np.isfinite(y).all():
            raise ValueError("validation calibration data must be finite")
        residual = y[..., None] - q
        weights = cluster_balanced_weights(clusters) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(weights) != n or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("sample_weights must be finite, nonnegative, and aligned")
        global_corr = _weighted_quantiles(residual, levels, weights)
        local, supports = {}, {}
        for key in np.unique(keys):
            rows = keys == key
            supports[_key(key)] = int(len(np.unique(clusters[rows & (weights > 0)])))
            local[_key(key)] = _weighted_quantiles(residual[rows], levels, weights[rows])
        positive = [count for count in supports.values() if count > 0]
        kappa = max(1.0, float(np.median(positive))) if positive else 1.0
        return cls(levels, global_corr, local, supports, kappa,
                   int(min_local_clusters), target_unit)

    def transform(self, predictions, context_key):
        q = np.asarray(predictions, dtype=np.float64)
        keys = np.asarray(context_key).reshape(-1)
        if q.shape[-1] != len(self.quantile_levels) or q.shape[0] != len(keys):
            raise ValueError("prediction shape or context keys are incompatible")
        output = q.copy()
        for i, key in enumerate(keys):
            key = _key(key)
            support = self.effective_support.get(key, 0)
            local = self.local_correction.get(key, self.global_correction)
            weight = 0.0 if support < self.min_local_clusters else support / (support + self.kappa_val)
            correction = (1.0 - weight) * self.global_correction + weight * local
            output[i] += correction
        return np.maximum.accumulate(output, axis=-1)

    def state_dict(self):
        return {
            "quantile_levels": self.quantile_levels.tolist(),
            "global_correction": self.global_correction.tolist(),
            "local_correction": {str(k): v.tolist() for k, v in self.local_correction.items()},
            "effective_support": {str(k): v for k, v in self.effective_support.items()},
            "kappa_val": self.kappa_val,
            "min_local_clusters": self.min_local_clusters,
            "target_unit": self.target_unit,
        }


def _key(value):
    item = value.item() if hasattr(value, "item") else value
    return int(item) if isinstance(item, (int, np.integer)) else str(item)


def _weighted_quantiles(values, levels, sample_weights):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    # Every independent cluster contributes total weight one, regardless of windows.
    weights = np.asarray(sample_weights, dtype=np.float64).copy()
    if len(weights) != n:
        raise ValueError("weights and values must align")
    if np.sum(weights) <= 0:
        raise ValueError("weighted quantile requires positive total weight")
    flat = values.reshape(n, -1, values.shape[-1])
    result = np.empty((flat.shape[1], len(levels)), dtype=np.float64)
    for output_index in range(flat.shape[1]):
        for q_index, level in enumerate(levels):
            order = np.argsort(flat[:, output_index, q_index], kind="stable")
            v, w = flat[order, output_index, q_index], weights[order]
            cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
            result[output_index, q_index] = np.interp(level, cumulative, v, left=v[0], right=v[-1])
    return result.reshape(values.shape[1:-1] + (len(levels),))


def cluster_balanced_weights(cluster_id):
    clusters = np.asarray(cluster_id).reshape(-1)
    _, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def probability_metrics(predictions, targets, quantile_levels=QUANTILE_LEVELS):
    q = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if q.shape[:-1] != y.shape or q.shape[-1] != len(levels):
        raise ValueError("predictions must have shape targets.shape + [Q]")
    error = y[..., None] - q
    pinball = np.maximum(levels * error, (levels - 1.0) * error)
    out = {"pinball": float(pinball.mean()), "crps": float(2.0 * pinball.mean())}
    for coverage in (0.5, 0.8, 0.9):
        tail = (1.0 - coverage) / 2.0
        lo = int(np.argmin(np.abs(levels - tail)))
        hi = int(np.argmin(np.abs(levels - (1.0 - tail))))
        out[f"coverage_{int(coverage * 100)}"] = float(((y >= q[..., lo]) & (y <= q[..., hi])).mean())
        out[f"width_{int(coverage * 100)}"] = float((q[..., hi] - q[..., lo]).mean())
    return out


def cluster_bootstrap(values, cluster_id, n_resamples=2000, seed=94017):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_id).reshape(-1)
    if values.shape[0] != len(clusters):
        raise ValueError("values and cluster_id must align on the first dimension")
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("at least two clusters are required for bootstrap")
    rng = np.random.RandomState(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    cluster_values = {cluster: values[clusters == cluster] for cluster in unique}
    for i in range(n_resamples):
        sample = rng.choice(unique, size=len(unique), replace=True)
        estimates[i] = np.concatenate([cluster_values[c] for c in sample], axis=0).mean()
    return {
        "estimate": float(values.mean()),
        "ci95": [float(v) for v in np.quantile(estimates, [0.025, 0.975])],
        "n_clusters": int(len(unique)),
    }
