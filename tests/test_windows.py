"""build_contiguous_windows 单元测试：计数公式、stride、空 split、越界保护。"""

import numpy as np

from mtkdg.data.common import build_contiguous_windows


def make_rows(n, c=3):
    values = np.arange(n * c, dtype=np.float32).reshape(n, c)
    times = np.arange(n, dtype=np.float32) * 10.0
    return values, times


def test_window_count_and_alignment():
    n, H, P, stride = 100, 10, 5, 3
    values, times = make_rows(n)
    s = build_contiguous_windows(values, times, H, P, stride, "train", row_offset=0)
    assert s.n_samples == (n - (H + P)) // stride + 1
    # 第一个窗口的 x/y 与源行精确对应
    assert np.allclose(s.x[0], values[:H])
    assert np.allclose(s.y_raw[0], values[H:H + P])
    assert np.allclose(s.t[0], times[:H])
    assert s.starts[0] == 0 and s.ends[0] == H + P
    assert s.starts[-1] == (s.n_samples - 1) * stride


def test_stride_one():
    n, H, P = 30, 10, 5
    values, times = make_rows(n)
    s = build_contiguous_windows(values, times, H, P, 1, "train", row_offset=0)
    assert s.n_samples == n - H - P + 1


def test_empty_when_rows_insufficient():
    values, times = make_rows(12)
    s = build_contiguous_windows(values, times, history_len=10, pred_len=5,
                                 stride=1, split_name="train", row_offset=0)
    assert s.n_samples == 0
    assert s.meta.get("empty_reason")


def test_row_offset_shifts_global_indices():
    values, times = make_rows(50)
    s = build_contiguous_windows(values, times, 10, 5, 10, "val", row_offset=1000)
    assert s.starts[0] == 1000
    assert s.ends[-1] <= 1000 + 50


def test_extra_condition_channel():
    n, H, P = 40, 10, 5
    values, times = make_rows(n)
    cond = (np.arange(n * 3, dtype=np.float32).reshape(n, 3) % 7) / 7.0
    s = build_contiguous_windows(values, times, H, P, 2, "train", 0,
                                 extra_channels={"condition": cond})
    assert s.condition is not None
    assert s.condition.shape == (s.n_samples, H, 3)
    assert np.allclose(s.condition[0], cond[:H])
