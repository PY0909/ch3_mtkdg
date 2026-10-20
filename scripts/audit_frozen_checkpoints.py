"""Read-only audit of a Chapter 3 point checkpoint for Chapter 4 reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtkdg.models.frozen_encoder import load_frozen_encoder


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(path: str, dataset: str) -> dict:
    encoder = load_frozen_encoder(path, dataset)
    checkpoint_sha256 = _sha256(path)
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(path)), "manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset") != dataset:
            raise ValueError(f"manifest dataset mismatch: {manifest.get('dataset')} != {dataset}")
        expected_prefix = manifest.get("checkpoint_identity", {}).get("best_sha256_16")
        if expected_prefix and not checkpoint_sha256.startswith(expected_prefix):
            raise ValueError("checkpoint SHA256 does not match the result manifest")
    return {
        "dataset": dataset,
        "checkpoint": os.path.abspath(path),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_fingerprint": encoder.fingerprint(),
        "encoder_frozen": all(not p.requires_grad for p in encoder.encoder.parameters()),
        "encoder_training": encoder.encoder.training,
        "metadata": encoder.metadata,
        "result_manifest": manifest_path if manifest is not None else None,
        "manifest_dataset": None if manifest is None else manifest.get("dataset"),
        "split_identity": None if manifest is None else manifest.get("split_identity"),
        "normalizer_identity": None if manifest is None else manifest.get("normalizer_identity"),
        "seed": None if manifest is None else manifest.get("seed"),
        "best_epoch": None if manifest is None else manifest.get("checkpoint_identity", {}).get("best_epoch"),
        "best_val_metric": None if manifest is None else manifest.get("checkpoint_identity", {}).get("best_val_RMSE"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=("fd002", "fd004", "metropt3"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = audit_checkpoint(args.checkpoint, args.dataset)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
