"""Round 1 评审修复项的回归测试（W1/W2/W3/W4/W6）。"""

import json
import os

import numpy as np
import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits, bundle_from_metropt_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.data.metropt import METROPT_MIN_STD, build_metropt_splits
from mtkdg.data.normalizer import TrainOnlyStandardScaler
from mtkdg.training.runner import TrainingRunner, set_seed


class TestW3VarianceFloor:
    def test_min_std_raises_floor(self):
        x = np.full((100, 2), 5.0)          # 常数通道
        x[:, 1] = np.random.RandomState(0).normal(0, 3, 100)
        scaler = TrainOnlyStandardScaler(min_std=0.01).fit(x)
        assert scaler.std[0] == pytest.approx(0.01)
        assert scaler.std[1] > 1.0
        # 常数通道 transform 后数值有界（不再是 std=1e-4 的 10^3 sigma）
        z = scaler.transform(np.array([[5.0, 0.0], [5.3, 1.0]]))
        assert np.abs(z[:, 0]).max() < 100.0

    def test_min_std_disabled_by_default(self):
        x = np.full((50, 1), 2.0)
        scaler = TrainOnlyStandardScaler().fit(x)
        assert scaler.std[0] < 0.001        # 保持旧行为（sqrt(0+eps)）

    def test_metropt_scaler_uses_floor(self, metropt_synthetic):
        s = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
        assert s.scaler.min_std == METROPT_MIN_STD
        assert (s.scaler.std >= METROPT_MIN_STD).all()
        assert s.meta["normalizer_min_std"] == METROPT_MIN_STD

    def test_identity_includes_min_std(self):
        x = np.random.RandomState(1).normal(0, 1, (40, 3))
        a = TrainOnlyStandardScaler(min_std=0.01).fit(x)
        b = TrainOnlyStandardScaler().fit(x)
        assert a.identity() != b.identity()

    def test_state_roundtrip_keeps_min_std(self):
        x = np.random.RandomState(2).normal(0, 1, (30, 4))
        s = TrainOnlyStandardScaler(min_std=0.01).fit(x)
        clone = TrainOnlyStandardScaler.from_state_dict(s.state_dict())
        assert clone.min_std == 0.01
        assert clone.identity() == s.identity()


class TestW4WindowMonths:
    def test_months_present_and_ordered(self, metropt_synthetic):
        s = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
        for split in (s.train, s.val, s.test):
            months = split.meta["window_months"]
            assert len(months) == split.n_samples
            assert months == sorted(months)
        assert s.val.meta["window_months"] == ["2020-02"] * s.val.n_samples
        assert set(m >= "2020-03" for m in s.test.meta["window_months"])


class TestW2UncappedLabels:
    def test_val_keeps_uncapped_labels(self, cmapss_synthetic):
        train_df, test_df, truth, _ = cmapss_synthetic
        s = build_cmapss_splits(
            data_root=None, subset="FD002", history_len=50, stride=2,
            training_cap=125, split_seed=2026,
            train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
        )
        for split in (s.train, s.val):
            unc = split.y_raw_uncapped
            assert unc is not None and unc.shape == split.y_raw.shape
            # capped 列 <= cap，uncapped 列可以超过
            assert (split.y_raw <= 125.0 + 1e-6).all()
            assert (unc >= split.y_raw - 1e-3).all()
            below = split.y_raw.reshape(-1) < 125.0 - 1e-3
            assert np.allclose(unc.reshape(-1)[below],
                               split.y_raw.reshape(-1)[below], atol=1e-3)
        assert s.test.y_raw_uncapped is None   # test 本来就是 raw


class TestW1SchedulerAndDiagnostic:
    @pytest.fixture(scope="class")
    def fd_bundle(self, cmapss_synthetic):
        train_df, test_df, truth, _ = cmapss_synthetic
        s = build_cmapss_splits(
            data_root=None, subset="FD002", history_len=50, stride=2,
            training_cap=125, split_seed=2026,
            train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
        )
        return bundle_from_cmapss_splits(s)

    def test_cosine_schedule_decays_lr(self, tmp_path, fd_bundle):
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=4, batch_size=64, device="cpu",
            output_dir=str(tmp_path), run_id="cosine", lr_scheduler="cosine", lr=1e-2,
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, fd_bundle)
        runner.fit(verbose=False)
        lrs = runner.history["lr"]
        assert len(lrs) == 4
        assert lrs[0] == pytest.approx(1e-2)
        assert lrs[-1] < lrs[0] * 0.6        # cosine 明显衰减
        assert all(b <= a for a, b in zip(lrs, lrs[1:]))   # 单调不增

    def test_history_contains_rawtarget_diagnostic(self, tmp_path, fd_bundle):
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=64, device="cpu",
            output_dir=str(tmp_path), run_id="rawdiag",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, fd_bundle)
        runner.fit(verbose=False)
        vals = runner.history["val_RMSE_rawtarget"]
        assert len(vals) == 1 and vals[0] == vals[0]      # 非 NaN
        # uncapped 目标的 RMSE 应 >= capped 目标的 RMSE（更多大目标难拟合时通常更大）
        assert vals[0] >= runner.history["val_RMSE"][0] - 1e-6

    def test_unknown_scheduler_raises(self, tmp_path, fd_bundle):
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, device="cpu", output_dir=str(tmp_path),
            run_id="bad", lr_scheduler="step",
        )
        with pytest.raises(ValueError):
            TrainingRunner(cfg, fd_bundle)


class TestW6Config:
    def test_rul_config_pred_len_null(self, cmapss_synthetic):
        train_df, test_df, truth, _ = cmapss_synthetic
        s = build_cmapss_splits(
            data_root=None, subset="FD002", history_len=50, stride=2,
            training_cap=125, split_seed=2026,
            train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
        )
        bundle = bundle_from_cmapss_splits(s)
        cfg = ExperimentConfig.for_dataset("fd002")
        d = cfg.to_dict()
        assert d["pred_len"] is None and d["task"] == "rul_regression"

    def test_state_config_pred_len_kept(self):
        cfg = ExperimentConfig.for_dataset("metropt3")
        assert cfg.to_dict()["pred_len"] == 24
