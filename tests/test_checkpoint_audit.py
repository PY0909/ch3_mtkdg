import json

from mtkdg.config import ExperimentConfig
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.training.checkpoint import save_checkpoint
from scripts.audit_frozen_checkpoints import audit_checkpoint


def test_checkpoint_audit_reports_protocol_fields(tmp_path):
    cfg = ExperimentConfig.for_dataset("fd002")
    cfg.model.d_model = 8
    model = MTKDGPointModel("rul_regression", 17, 0, cfg.model, rul_mean=60.0, rul_std=20.0)
    path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(str(path), model, epoch=4, best_metric=1.2, extra={"config": cfg.to_dict()})

    report = audit_checkpoint(str(path), dataset="fd002")

    assert report["dataset"] == "fd002"
    assert report["encoder_fingerprint"]
    assert report["checkpoint_sha256"]
    assert report["encoder_frozen"] is True
