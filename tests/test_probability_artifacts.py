import json

import numpy as np
import pytest

from mtkdg.evaluation.probability_artifacts import save_probability_artifacts


def test_artifacts_roundtrip_and_refuse_overwrite(tmp_path):
    run_dir = tmp_path / "run1"
    payload = {
        "dataset": "fd002",
        "training_target": "raw",
        "encoder_fingerprint": "abc123",
        "split_identity": "split-v1",
        "scaler_identity": "scale-v1",
        "quality_gate_status": "pending",
        "evaluated_once": False,
    }
    paths = save_probability_artifacts(
        str(run_dir), payload,
        arrays={"val_targets_raw": np.arange(3), "val_quantiles": np.zeros((3, 19))},
    )
    assert json.loads(open(paths["manifest"], encoding="utf-8").read())["encoder_fingerprint"] == "abc123"
    assert np.load(paths["arrays"])["val_targets_raw"].shape == (3,)
    with pytest.raises(FileExistsError):
        save_probability_artifacts(str(run_dir), payload, arrays={})


def test_artifacts_require_provenance_and_test_marker(tmp_path):
    with pytest.raises(ValueError, match="missing required fields"):
        save_probability_artifacts(str(tmp_path / "bad"), {"dataset": "fd002"}, arrays={})
