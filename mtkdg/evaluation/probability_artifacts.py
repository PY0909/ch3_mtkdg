"""Auditable persistence for contextual probability experiments."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

import numpy as np


REQUIRED_FIELDS = {
    "dataset", "training_target", "encoder_fingerprint", "split_identity",
    "scaler_identity", "quality_gate_status", "evaluated_once",
}


def save_probability_artifacts(run_dir: str, manifest: Dict, arrays: Dict[str, np.ndarray]):
    missing = REQUIRED_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if os.path.exists(run_dir):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    os.makedirs(run_dir, exist_ok=False)
    array_path = os.path.join(run_dir, "probability_arrays.npz")
    clean_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    for key, value in clean_arrays.items():
        if value.dtype.kind in "biufc" and not np.isfinite(value).all():
            raise ValueError(f"array {key} contains non-finite values")
    np.savez_compressed(array_path, **clean_arrays)
    manifest = dict(manifest)
    manifest["arrays_file"] = os.path.basename(array_path)
    manifest["arrays_sha256"] = _sha256(array_path)
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return {"manifest": manifest_path, "arrays": array_path}


def load_probability_artifacts(run_dir: str):
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    array_path = os.path.join(run_dir, manifest["arrays_file"])
    if _sha256(array_path) != manifest["arrays_sha256"]:
        raise ValueError("probability array hash mismatch")
    arrays = dict(np.load(array_path, allow_pickle=False))
    return manifest, arrays


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
