import numpy as np

from mtkdg.data.calibration import make_calibration_view
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.data.metropt import build_metropt_splits


def test_fd_view_uses_one_last_window_per_validation_engine(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    splits = build_cmapss_splits(
        None, "FD002", history_len=50, stride=1, train_df=train_df.copy(),
        test_df=test_df.copy(), test_truth=truth,
    )
    view = make_calibration_view(splits.val, task="rul_regression")

    assert view.n_samples == len(np.unique(splits.val.meta["engine_ids"]))
    assert len(np.unique(view.meta["cluster_id"])) == view.n_samples


def test_metropt_view_has_nonoverlap_targets_and_natural_week_clusters(metropt_synthetic):
    splits = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=2)
    view = make_calibration_view(splits.val, task="state_forecast", pred_len=8)

    assert view.n_samples > 0
    assert np.all(np.diff(view.starts) >= 8)
    assert all(len(str(value)) == 7 for value in view.meta["cluster_id"])
