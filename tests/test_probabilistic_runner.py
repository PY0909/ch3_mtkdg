import numpy as np
import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.training.checkpoint import save_checkpoint
from mtkdg.training.probabilistic_runner import ContextualProbabilisticRunner
from mtkdg.data.bundle import bundle_from_metropt_splits
from mtkdg.data.metropt import build_metropt_splits


def test_runner_trains_only_readout_calibrates_validation_and_guards_test(tmp_path, cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    splits = build_cmapss_splits(
        None, "FD002", history_len=50, stride=2, training_target="raw",
        train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
    )
    bundle = bundle_from_cmapss_splits(splits)
    cfg = ExperimentConfig.for_dataset("fd002")
    cfg.model.d_model = 8
    point_model = MTKDGPointModel("rul_regression", 17, 0, cfg.model,
                                  rul_mean=bundle.rul_mean, rul_std=bundle.rul_std)
    cfg_path = tmp_path / "point.pt"
    save_checkpoint(str(cfg_path), point_model, epoch=3, best_metric=1.0,
                    extra={"config": cfg.to_dict()})
    runner = ContextualProbabilisticRunner(
        bundle=bundle, checkpoint_path=str(cfg_path), dataset="fd002",
        seed=2026, mode="C1", epochs=1, batch_size=16, device="cpu",
    )

    history = runner.fit()
    val = runner.calibrate()
    test = runner.evaluate_test_once()

    assert len(history) == 1
    assert val["quantiles"].shape[0] == len(np.unique(splits.val.meta["engine_ids"]))
    assert test["quantiles"].shape == (splits.test.n_samples, len(runner.quantile_levels))
    with pytest.raises(RuntimeError, match="once"):
        runner.evaluate_test_once()
    assert all(p.grad is None for p in runner.encoder.encoder.parameters())


def test_runner_supports_metropt_multihorizon_cpu_path(tmp_path, metropt_synthetic):
    splits = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
    bundle = bundle_from_metropt_splits(splits)
    cfg = ExperimentConfig.for_dataset("metropt3", history_len=24, pred_len=8, stride=6)
    cfg.model.d_model = 8
    point_model = MTKDGPointModel("state_forecast", 15, 8, cfg.model)
    cfg_path = tmp_path / "metropt_point.pt"
    save_checkpoint(str(cfg_path), point_model, epoch=3, best_metric=1.0,
                    extra={"config": cfg.to_dict()})
    runner = ContextualProbabilisticRunner(
        bundle=bundle, checkpoint_path=str(cfg_path), dataset="metropt3",
        seed=2026, mode="C1", epochs=1, batch_size=8, device="cpu",
    )

    runner.fit()
    val = runner.calibrate()
    test = runner.evaluate_test_once()

    assert val["quantiles"].shape[1:] == (8, 15, len(runner.quantile_levels))
    assert test["quantiles"].shape[1:] == (8, 15, len(runner.quantile_levels))
    assert all(p.grad is None for p in runner.encoder.encoder.parameters())
