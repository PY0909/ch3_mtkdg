"""MetroPT-3 首月协议测试：边界定义、split 标签、窗口不跨界、train-only 归一化。"""

import numpy as np
import pandas as pd
import pytest

from mtkdg.data.metropt import (
    METROPT_SENSOR_COLUMNS,
    SPLIT_IDENTITY,
    assign_split_labels,
    build_metropt_splits,
    first_month_boundaries,
    load_metropt_frame,
)


@pytest.fixture(scope="module")
def splits(metropt_synthetic):
    return build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)


class TestFirstMonthBoundaries:
    def test_month_window(self, metropt_synthetic):
        b = first_month_boundaries(metropt_synthetic["timestamp"])
        assert b["month_start"] == pd.Timestamp("2020-02-01 00:00:00")
        assert b["month_end"] == pd.Timestamp("2020-03-01 00:00:00")  # 闰年 29 天
        expected = pd.Timestamp("2020-02-01") + pd.Timedelta(seconds=0.8 * 29 * 86400)
        assert b["train_val_ts"] == expected

    def test_split_label_counts(self, metropt_synthetic):
        labels = assign_split_labels(metropt_synthetic)
        n = len(labels)
        n_train = (labels == "train").sum()
        n_val = (labels == "val").sum()
        n_test = (labels == "test").sum()
        # 合成数据共 45 天：2 月 29 天（闰年）+ 3 月 16 天
        # train = 2 月前 80% 时间 ≈ 23.2 天；val = 2 月剩余 ≈ 5.8 天；test = 3 月全部
        expected_train = 0.8 * 29 * 86400 / 300
        assert abs(n_train - expected_train) <= 2
        assert abs(n_val - (29 * 86400 / 300 - n_train)) <= 2
        assert n_test == int(16 * 86400 / 300) + 1
        assert n_train + n_val + n_test == n
        # test 必须从 2020-03-01 开始
        ts = pd.DatetimeIndex(metropt_synthetic["timestamp"])
        assert ts[labels == "test"][0] >= pd.Timestamp("2020-03-01")
        assert ts[labels == "val"].max() < pd.Timestamp("2020-03-01")


class TestWindowsDoNotCrossSplits:
    def test_window_containment(self, splits, metropt_synthetic):
        """每个窗口 [start, end) 必须完整落在所属 split 的行范围内。"""
        labels = assign_split_labels(metropt_synthetic)
        # 每个 split 的全局行范围
        ranges = {}
        for name in ("train", "val", "test"):
            idx = np.flatnonzero(labels == name)
            ranges[name] = (idx[0], idx[-1] + 1)

        arr = {"train": splits.train, "val": splits.val, "test": splits.test}
        for name, split in arr.items():
            lo, hi = ranges[name]
            assert (split.starts >= lo).all(), f"{name} window starts before split begins"
            assert (split.ends <= hi).all(), f"{name} window ends after split ends"
            # 交叉验证：窗口覆盖的任何一行都属于本 split
            for s, e in [(int(split.starts[0]), int(split.ends[0])),
                         (int(split.starts[-1]), int(split.ends[-1]))]:
                assert set(labels[s:e]) == {name}, f"{name} window crosses boundary"

    def test_no_val_window_reads_train_rows(self, splits, metropt_synthetic):
        labels = assign_split_labels(metropt_synthetic)
        for s, e in zip(splits.val.starts, splits.val.ends):
            assert labels[s:e].tolist() == ["val"] * (e - s)

    def test_no_test_window_reads_first_month(self, splits, metropt_synthetic):
        """测试窗口不能回看首月末尾的观测。"""
        labels = assign_split_labels(metropt_synthetic)
        for s, e in zip(splits.test.starts, splits.test.ends):
            assert "train" not in set(labels[s:e])
            assert "val" not in set(labels[s:e])


class TestMetroPTProtocol:
    def test_shapes(self, splits):
        H, P, C = 24, 8, len(METROPT_SENSOR_COLUMNS)
        for split in (splits.train, splits.val, splits.test):
            assert split.x.shape[1:] == (H, C)
            assert split.y.shape[1:] == (P, C)
            assert split.mask.shape == split.x.shape
            assert np.isfinite(split.x).all()
            assert np.isfinite(split.y).all()

    def test_window_count_formula(self, splits, metropt_synthetic):
        labels = assign_split_labels(metropt_synthetic)
        H, P, stride = 24, 8, 6
        for name, split in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            idx = np.flatnonzero(labels == name)
            n_rows = len(idx)
            expected = max(0, (n_rows - (H + P)) // stride + 1)
            assert split.n_samples == expected, name

    def test_train_only_normalization(self, splits, metropt_synthetic):
        """scaler 统计量必须只来自 train 时间段。"""
        labels = assign_split_labels(metropt_synthetic)
        raw = metropt_synthetic[METROPT_SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        train_rows = raw[labels == "train"]
        assert np.allclose(splits.scaler.mean, train_rows.mean(axis=0), atol=1e-5)
        assert np.allclose(splits.scaler.std ** 2, train_rows.var(axis=0), atol=1e-3)
        # 已标准化的 train x 均值应接近 0；test x 不应被重新中心化
        assert np.abs(splits.train.x.mean(axis=(0, 1))).max() < 0.05

    def test_y_raw_kept_in_original_units(self, splits, metropt_synthetic):
        """y_raw 必须等于对应原始行（未标准化），供 raw-unit 指标使用。"""
        raw = metropt_synthetic[METROPT_SENSOR_COLUMNS].to_numpy(dtype=np.float32)
        split = splits.train
        s0, e0 = int(split.starts[0]), int(split.ends[0])
        expected_target = raw[s0 + 24:e0]
        assert np.allclose(split.y_raw[0], expected_target, atol=1e-4)

    def test_split_identity_recorded(self, splits):
        assert splits.meta["split_identity"] == SPLIT_IDENTITY
        assert len(splits.identity) == 16
        assert "train_val_ts" in splits.boundaries


class TestLoadFrame:
    def test_load_synthetic_csv(self, tmp_path, metropt_synthetic):
        csv = tmp_path / "metro.csv"
        metropt_synthetic.index.name = "Unnamed: 0"
        metropt_synthetic.to_csv(csv, index=True)
        frame = load_metropt_frame(str(csv))
        assert list(frame.columns[:1]) == ["timestamp"]
        assert (frame["delta_t"].iloc[1:] == 300).all()
        assert frame["t_rel"].is_monotonic_increasing
