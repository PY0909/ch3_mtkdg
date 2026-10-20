"""C-MAPSS FD002/FD004 adapter 测试（合成数据）。

覆盖：engine-level split、窗口不跨 engine split、capped/raw 标签、末端窗口、
左填充、train-only 归一化、FD002 与 FD004 独立性。
"""

import numpy as np
import pytest

from mtkdg.data.cmapss import (
    DROPPED_SENSORS_FD002,
    build_cmapss_splits,
    engine_level_split,
    retained_channels,
    split_identity_for,
)

H = 50
CAP = 125


@pytest.fixture(scope="module")
def build(cmapss_synthetic):
    train_df, test_df, truth, lengths = cmapss_synthetic

    def _build(subset, **kw):
        return build_cmapss_splits(
            data_root=None, subset=subset, history_len=H, stride=1,
            training_cap=CAP, split_seed=2026,
            train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
            **kw,
        )

    return _build


class TestEngineLevelSplit:
    def test_disjoint_and_deterministic(self):
        ids = list(range(1, 21))
        tr1, va1 = engine_level_split(ids, val_ratio=0.2, seed=2026)
        tr2, va2 = engine_level_split(ids, val_ratio=0.2, seed=2026)
        assert tr1 == tr2 and va1 == va2
        assert set(tr1) | set(va1) == set(ids)
        assert not (set(tr1) & set(va1))
        assert tr1 == sorted(tr1) and va1 == sorted(va1)

    def test_seed_changes_split(self):
        ids = list(range(1, 21))
        _, va1 = engine_level_split(ids, val_ratio=0.2, seed=2026)
        _, va2 = engine_level_split(ids, val_ratio=0.2, seed=2027)
        assert va1 != va2


class TestChannels:
    def test_retained_channels(self):
        ch = retained_channels("FD002")
        assert len(ch) == 17
        assert ch[:3] == ["setting1", "setting2", "setting3"]
        for dropped in DROPPED_SENSORS_FD002:
            assert dropped not in ch
        assert retained_channels("FD004") == retained_channels("FD002")


class TestWindowsAndLabels:
    def test_engine_isolation(self, build):
        """同一 engine 的窗口不得跨 train/validation。"""
        s = build("FD002")
        train_eng = set(s.train.meta["engine_ids"].tolist())
        val_eng = set(s.val.meta["engine_ids"].tolist())
        assert train_eng and val_eng
        assert not (train_eng & val_eng)
        assert train_eng == set(s.train_engines)
        assert val_eng == set(s.val_engines)

    def test_train_labels_capped(self, build, cmapss_synthetic):
        train_df = cmapss_synthetic[0]
        s = build("FD002")
        y_raw = s.train.y_raw.reshape(-1)
        # 标签来自窗口末端 cycle 的 raw RUL，且被 cap
        assert (y_raw >= 0).all()
        assert (y_raw <= CAP).all()
        # 手工核对第一个 engine 的第一个窗口
        eng = s.train_engines[0]
        rows = train_df[train_df["id"] == eng]
        first_label = min(float(rows["cycle"].max() - H), CAP)
        # train split 中该 engine 的第一个窗口标签
        eng_positions = np.flatnonzero(s.train.meta["engine_ids"] == eng)
        assert np.isclose(s.train.y_raw[eng_positions[0], 0], first_label)

    def test_test_labels_official_raw(self, build, cmapss_synthetic):
        """test 标签必须等于官方 RUL 文件（raw，不 cap，可超过 125）。"""
        _, _, truth, _ = cmapss_synthetic
        s = build("FD002")
        y = s.test.y_raw.reshape(-1)
        assert np.allclose(np.sort(y), np.sort(truth))
        assert (y > CAP).any(), "合成官方 RUL 应包含 > cap 的值以验证不截断"

    def test_one_window_per_test_engine_and_padding(self, build, cmapss_synthetic):
        _, test_df, _, lengths = cmapss_synthetic
        s = build("FD002")
        assert s.test.n_samples == len(lengths)
        engines = s.test.meta["engine_ids"]
        assert len(np.unique(engines)) == len(engines)
        # 短 engine 左填充：x 前 pad 行应重复首观测
        for i, L in enumerate(lengths):
            pad = max(0, H - L)
            if pad:
                assert np.allclose(s.test.x[i, :pad], s.test.x[i, pad])
        assert s.test.meta["n_padded"] == sum(1 for L in lengths if L < H)

    def test_test_window_is_last(self, build, cmapss_synthetic):
        _, test_df, _, _ = cmapss_synthetic
        s = build("FD004")
        for i, eng in enumerate(s.test.meta["engine_ids"]):
            rows = test_df[test_df["id"] == eng]
            L = len(rows)
            # 末端窗口时间坐标的最后一位 = 该 engine 最后 cycle
            assert s.test.t[i, -1] == rows["cycle"].max()
            if L >= H:
                # 无填充时 t 是最后 H 个 cycle
                assert np.allclose(s.test.t[i], rows["cycle"].to_numpy()[-H:])

    def test_train_only_feature_normalization(self, build, cmapss_synthetic):
        train_df = cmapss_synthetic[0]
        s = build("FD002")
        ch = retained_channels("FD002")
        train_rows = train_df[train_df["id"].isin(s.train_engines)][ch].to_numpy(dtype=np.float64)
        assert np.allclose(s.feature_scaler.mean, train_rows.mean(axis=0), atol=1e-6)
        # val/test 用同一 scaler；被删通道的审计存在
        assert set(s.meta["dropped_sensor_std_train"]) == set(DROPPED_SENSORS_FD002)
        assert all(v < 1e-6 for v in s.meta["dropped_sensor_std_train"].values())

    def test_target_scaler_from_train_capped(self, build):
        s = build("FD002")
        m = s.train.y_raw.mean()
        sd = s.train.y_raw.std()
        assert np.isclose(s.target_scaler.mean[0], m, atol=1e-4)
        assert np.isclose(s.target_scaler.std[0], sd, atol=1e-3)


class TestIndependence:
    def test_fd002_fd004_separate_identity(self, build):
        a = build("FD002")
        b = build("FD004")
        assert a.identity != b.identity
        assert a.split_identity == split_identity_for("FD002")
        assert b.split_identity == split_identity_for("FD004")

    def test_shapes(self, build):
        for subset in ("FD002", "FD004"):
            s = build(subset)
            for split in (s.train, s.val, s.test):
                assert split.x.shape[1:] == (H, 17)
                assert split.y.shape[1:] == (1,)
                assert split.condition.shape[1:] == (H, 3)
                assert split.mask.shape == split.x.shape
                assert np.isfinite(split.x).all()
            # 官方 test engine 数 = truth 条目数
            assert s.test.n_samples == 5
