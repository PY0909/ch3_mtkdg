import copy

import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.models.frozen_encoder import FrozenEncoder, load_frozen_encoder
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.training.checkpoint import save_checkpoint


def _checkpoint(tmp_path):
    cfg = ExperimentConfig.for_dataset("fd002")
    cfg.model.d_model = 8
    model = MTKDGPointModel("rul_regression", 17, 0, cfg.model, rul_mean=60.0, rul_std=20.0)
    path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(str(path), model, epoch=4, best_metric=1.2, extra={"config": cfg.to_dict()})
    return path, cfg, model


def _batch(batch_size=2):
    x = torch.randn(batch_size, 50, 17)
    t = torch.arange(50.0).unsqueeze(0).expand(batch_size, -1).contiguous()
    cond = torch.randn(batch_size, 50, 3)
    return x, t, torch.ones_like(x), cond


def test_loader_freezes_encoder_and_preserves_forward_contract(tmp_path):
    path, _, _ = _checkpoint(tmp_path)
    frozen = load_frozen_encoder(str(path), dataset="fd002")
    x, t, mask, cond = _batch()

    out = frozen(x, t, mask, cond)
    assert out["z"].shape == (2, 17, 8)
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    assert frozen.encoder.training is False
    assert frozen.fingerprint()


def test_train_call_cannot_reenable_encoder_training_mode(tmp_path):
    path, _, _ = _checkpoint(tmp_path)
    frozen = load_frozen_encoder(str(path), dataset="fd002")

    frozen.train()

    assert frozen.training is True
    assert frozen.encoder.training is False


def test_optimizer_step_does_not_change_encoder_fingerprint(tmp_path):
    path, _, _ = _checkpoint(tmp_path)
    frozen = load_frozen_encoder(str(path), dataset="fd002")
    before = frozen.fingerprint()
    trainable = torch.nn.Linear(8, 1)
    optimizer = torch.optim.SGD(trainable.parameters(), lr=0.1)
    x, t, mask, cond = _batch()
    out = frozen(x, t, mask, cond)["z"].mean(dim=1)
    loss = trainable(out).square().mean()
    loss.backward()
    optimizer.step()

    assert frozen.fingerprint() == before
    assert all(p.grad is None for p in frozen.encoder.parameters())


def test_loader_rejects_wrong_dataset_and_missing_encoder_keys(tmp_path):
    path, _, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="dataset"):
        load_frozen_encoder(str(path), dataset="fd004")

    payload = torch.load(path, weights_only=False)
    payload["state_dict"].pop("encoder.out_norm.weight")
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="encoder"):
        load_frozen_encoder(str(broken), dataset="fd002")


def test_fingerprint_changes_when_encoder_parameter_changes(tmp_path):
    path, _, _ = _checkpoint(tmp_path)
    frozen = load_frozen_encoder(str(path), dataset="fd002")
    before = frozen.fingerprint()
    with torch.no_grad():
        next(frozen.encoder.parameters()).add_(0.01)
    assert frozen.fingerprint() != before
