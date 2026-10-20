"""共享测试夹具：合成 MetroPT 与合成 C-MAPSS 数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtkdg.data.metropt import METROPT_SENSOR_COLUMNS


def make_metropt_frame(n_days: int = 45, interval_s: int = 300, seed: int = 7):
    """合成 MetroPT 帧：2020-02-01 起覆盖超过两个自然月，15 传感器随时间变化。"""
    rng = np.random.RandomState(seed)
    n = int(n_days * 86400 / interval_s) + 1
    ts = pd.date_range("2020-02-01 00:00:00", periods=n, freq=f"{interval_s}s")
    frame = pd.DataFrame({"timestamp": ts})
    for i, col in enumerate(METROPT_SENSOR_COLUMNS):
        base = np.arange(n, dtype=np.float64)
        frame[col] = (0.5 * (i + 1) + 0.001 * (i + 1) * base
                      + rng.normal(0, 0.01, n))
    return frame


def make_cmapss_frames(
    n_train_engines: int = 12,
    min_len: int = 60,
    max_len: int = 200,
    n_test_engines: int = 5,
    test_min_len: int = 30,
    test_max_len: int = 80,
    seed: int = 11,
):
    """合成 C-MAPSS train/test/RUL 三件套。

    - s1,s5,s6,s10,s16,s18,s19 为近常数（应被删除）；
    - 其余传感器随 cycle 退化；
    - test 官方 RUL 部分超过 125（验证 test 不 cap）。
    """
    from mtkdg.data.cmapss import ALL_SENSOR_COLUMNS, CMAPSS_COLUMNS

    rng = np.random.RandomState(seed)
    # 3 个离散工况组合（与真实 FD002/FD004 的离散工况结构一致，供 V1 消融测试）
    cond_grid = [(0.0, 0.0, 100.0), (10.0, 0.25, 90.0), (20.0, 0.5, 80.0)]
    rows = []
    for eng in range(1, n_train_engines + 1):
        L = rng.randint(min_len, max_len + 1)
        for cyc in range(1, L + 1):
            row = {"id": eng, "cycle": cyc}
            for si, s in enumerate(SETTING_SOURCES):
                row[s] = float(cond_grid[eng % 3][si])
            for si, s in enumerate(ALL_SENSOR_COLUMNS):
                if s in ("s1", "s5", "s6", "s10", "s16", "s18", "s19"):
                    row[s] = 100.0
                else:
                    row[s] = 5.0 + 0.02 * cyc * ((si % 5) + 1) + 0.05 * rng.randn()
            rows.append(row)
    train_df = pd.DataFrame(rows)[CMAPSS_COLUMNS]

    rows = []
    lengths = []
    for eng in range(1, n_test_engines + 1):
        L = rng.randint(test_min_len, test_max_len + 1)
        lengths.append(L)
        for cyc in range(1, L + 1):
            row = {"id": eng, "cycle": cyc}
            for si, s in enumerate(SETTING_SOURCES):
                row[s] = float(cond_grid[eng % 3][si])
            for si, s in enumerate(ALL_SENSOR_COLUMNS):
                if s in ("s1", "s5", "s6", "s10", "s16", "s18", "s19"):
                    row[s] = 100.0
                else:
                    row[s] = 5.0 + 0.02 * cyc * ((si % 5) + 1) + 0.05 * rng.randn()
            rows.append(row)
    test_df = pd.DataFrame(rows)[CMAPSS_COLUMNS]
    # 官方 RUL：部分 > 125，验证 test 用 raw、train/val 用 capped
    truth = np.array([80.0, 130.0, 45.0, 160.0, 125.0][:n_test_engines], dtype=np.float64)
    return train_df, test_df, truth, lengths


SETTING_SOURCES = ["setting1", "setting2", "setting3"]


@pytest.fixture(scope="session")
def metropt_synthetic():
    return make_metropt_frame()


@pytest.fixture(scope="session")
def cmapss_synthetic():
    return make_cmapss_frames()
