"""真实数据测试：MetroPT 首月边界、FD002/FD004 adapter、CPU one-batch 训练 smoke。

依赖 /home/new_work/kaf_profiti/dataset 下的原始文件；文件缺失时整体跳过。
"""

import os

import numpy as np
import pytest
import torch

DATA_ROOT = "/home/new_work/kaf_profiti/dataset"
METROPT_CSV = os.path.join(DATA_ROOT, "metropt+3+dataset", "MetroPT3(AirCompressor).csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_ROOT), reason=f"dataset root not found: {DATA_ROOT}"
)


@pytest.fixture(scope="module")
def metropt_real_splits():
    from mtkdg.data.metropt import build_metropt_splits, load_metropt_frame

    frame = load_metropt_frame(METROPT_CSV)
    return build_metropt_splits(frame)  # 默认 H=168 P=24 stride=60


@pytest.fixture(scope="module")
def fd_real_splits(request):
    from mtkdg.data.cmapss import build_cmapss_splits

    subset = request.param
    return build_cmapss_splits(data_root=DATA_ROOT, subset=subset)  # 默认 H=50


class TestMetroPTReal:
    def test_first_month_boundaries(self, metropt_real_splits):
        b = metropt_real_splits.boundaries
        assert b["month_start"] == "2020-02-01 00:00:00"
        assert b["month_end"] == "2020-03-01 00:00:00"
        # 2020 闰年：2 月 29 天，train 边界 = 2/1 + 0.8*29d = 2/24 04:48
        assert b["train_val_ts"] == "2020-02-24 04:48:00"
        meta = metropt_real_splits.meta
        assert meta["n_rows_total"] == 1516948

    def test_window_counts_and_containment(self, metropt_real_splits):
        s = metropt_real_splits
        # 2 月共 214850 行：train ≈ 0.8*29d ≈ 171k，val ≈ 43k
        tr_rows = s.boundaries["train_rows"]
        va_rows = s.boundaries["val_rows"]
        te_rows = s.boundaries["test_rows"]
        assert 160_000 < tr_rows[1] - tr_rows[0] < 185_000
        assert 30_000 < va_rows[1] - va_rows[0] < 60_000
        assert te_rows[1] - te_rows[0] > 1_000_000
        # 窗口落在各自行范围内（不跨边界）
        assert s.train.starts.min() >= tr_rows[0] and s.train.ends.max() <= tr_rows[1]
        assert s.val.starts.min() >= va_rows[0] and s.val.ends.max() <= va_rows[1]
        assert s.test.starts.min() >= te_rows[0] and s.test.ends.max() <= te_rows[1]
        # 默认协议的样本量级
        assert 2000 < s.train.n_samples < 4000
        assert 500 < s.val.n_samples < 1200
        assert s.test.n_samples > 15_000

    def test_gap_statistics_recorded(self, metropt_real_splits):
        stats = metropt_real_splits.meta["gap_stats_all"]
        assert stats["n_gaps_gt_600s"] > 0          # 真实存在长断档
        assert stats["max_delta_s"] > 100_000       # 最长约 2 天
        assert stats["median_delta_s"] == pytest.approx(10.0)

    def test_arrays_finite(self, metropt_real_splits):
        s = metropt_real_splits
        for split in (s.train, s.val, s.test):
            assert np.isfinite(split.x).all()
            assert np.isfinite(split.y).all()
            assert split.x.shape[1:] == (168, 15)
            assert split.y.shape[1:] == (24, 15)


@pytest.mark.parametrize("fd_real_splits", ["FD002", "FD004"], indirect=True)
class TestCMAPSSReal:
    EXPECT = {"FD002": (260, 259), "FD004": (249, 248)}  # (train engines, test engines)

    def test_engine_counts(self, fd_real_splits):
        s = fd_real_splits
        subset = s.subset
        n_train_total, n_test = self.EXPECT[subset]
        assert s.meta["n_train_engines"] + s.meta["n_val_engines"] == n_train_total
        assert s.test.n_samples == n_test
        assert s.meta["n_test_engines"] == n_test

    def test_engine_isolation(self, fd_real_splits):
        s = fd_real_splits
        train_eng = set(s.train.meta["engine_ids"].tolist())
        val_eng = set(s.val.meta["engine_ids"].tolist())
        assert not (train_eng & val_eng)
        assert train_eng == set(s.train_engines)

    def test_test_labels_official_raw(self, fd_real_splits):
        s = fd_real_splits
        truth = np.loadtxt(os.path.join(DATA_ROOT, "CMAPSSData", f"RUL_{s.subset}.txt"))
        y = s.test.y_raw.reshape(-1)
        assert np.allclose(y, truth, atol=1e-4)
        assert y.max() > 125.0          # raw 不截断

    def test_train_val_labels_capped(self, fd_real_splits):
        s = fd_real_splits
        assert s.train.y_raw.max() <= 125.0 + 1e-6
        assert s.val.y_raw.max() <= 125.0 + 1e-6
        assert s.train.y_raw.min() >= 0

    def test_shapes_finite(self, fd_real_splits):
        s = fd_real_splits
        for split in (s.train, s.val, s.test):
            assert split.x.shape[1:] == (50, 17)
            assert split.condition.shape[1:] == (50, 3)
            assert np.isfinite(split.x).all()
        assert 30_000 < s.train.n_samples < 60_000


@pytest.mark.parametrize("fd_real_splits", ["FD002"], indirect=True)
class TestRealOneBatchSmoke:
    def test_fd002_one_batch_train_step(self, fd_real_splits):
        """真实 FD002：单个 batch 前向+反向+优化器步进，数值全部有限。"""
        from mtkdg.config import ExperimentConfig
        from mtkdg.data.bundle import bundle_from_cmapss_splits
        from mtkdg.models.heads import MTKDGPointModel
        from mtkdg.training.runner import set_seed

        bundle = bundle_from_cmapss_splits(fd_real_splits)
        cfg = ExperimentConfig.for_dataset("fd002")
        cfg.model.d_model = 32
        set_seed(2026)
        model = MTKDGPointModel(
            "rul_regression", bundle.n_channels, 0, cfg.model,
            rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        split = bundle.train
        idx = np.arange(64)
        batch = {
            "x": torch.from_numpy(split.x[idx]),
            "t": torch.from_numpy(split.t[idx]),
            "mask": torch.from_numpy(split.mask[idx]),
            "y": torch.from_numpy(split.y[idx]),
            "condition": torch.from_numpy(split.condition[idx]),
        }
        pred = model(batch["x"], batch["t"], batch["mask"], batch["condition"])
        assert pred.shape == (64, 1) and (pred >= 0).all()
        loss = ((pred - bundle.rul_mean) / bundle.rul_std - batch["y"]).pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss)
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        assert grads_finite
        opt.step()
        with torch.no_grad():
            pred2 = model(batch["x"], batch["t"], batch["mask"], batch["condition"])
        assert torch.isfinite(pred2).all()


def test_metropt_one_batch_train_step(metropt_real_splits):
    """真实 MetroPT：单个 batch 前向+反向+优化器步进。"""
    from mtkdg.config import ExperimentConfig
    from mtkdg.data.bundle import bundle_from_metropt_splits
    from mtkdg.models.heads import MTKDGPointModel
    from mtkdg.training.runner import set_seed

    bundle = bundle_from_metropt_splits(metropt_real_splits)
    cfg = ExperimentConfig.for_dataset("metropt3")
    cfg.model.d_model = 32
    set_seed(2026)
    model = MTKDGPointModel("state_forecast", bundle.n_channels, 24, cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    split = bundle.train
    idx = np.arange(32)
    x = torch.from_numpy(split.x[idx])
    t = torch.from_numpy(split.t[idx])
    y = torch.from_numpy(split.y[idx])
    pred = model(x, t, None, None)
    assert pred.shape == (32, 24, 15)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert torch.isfinite(loss)
    opt.step()
    with torch.no_grad():
        pred2 = model(x, t, None, None)
    assert torch.isfinite(pred2).all()
