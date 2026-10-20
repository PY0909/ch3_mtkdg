import numpy as np
import pytest

from mtkdg.evaluation.calibration import (
    ShrunkQuantileCalibrator,
    probability_metrics,
)


def _arrays(n=12):
    rng = np.random.RandomState(3)
    y = rng.normal(size=(n, 2))
    q = np.stack([y - 1.0, y - 0.5, y, y + 0.5, y + 1.0], axis=-1)
    keys = np.repeat(np.arange(2), n // 2)
    clusters = np.arange(n)
    return q, y, keys, clusters, (0.05, 0.25, 0.5, 0.75, 0.95)


def test_calibrator_requires_validation_calibration_split():
    q, y, keys, clusters, levels = _arrays()
    with pytest.raises(ValueError, match="val_calibration"):
        ShrunkQuantileCalibrator.fit(q, y, keys, clusters, split="test")


def test_low_cluster_support_falls_back_to_global_and_keeps_monotonicity():
    q, y, keys, clusters, levels = _arrays()
    clusters[:] = np.repeat(np.arange(3), 4)
    calibrator = ShrunkQuantileCalibrator.fit(
        q, y, keys, clusters, split="val_calibration", quantile_levels=levels, min_local_clusters=5
    )
    corrected = calibrator.transform(q, keys)

    expected = np.maximum.accumulate(q + calibrator.global_correction[None, ...], axis=-1)
    assert np.allclose(corrected, expected)
    assert (corrected[..., 1:] >= corrected[..., :-1]).all()


def test_cluster_weighted_local_calibrator_uses_cluster_count_not_window_count():
    q, y, keys, clusters, levels = _arrays()
    clusters = np.repeat(np.arange(6), 2)
    calibrator = ShrunkQuantileCalibrator.fit(
        q, y, keys, clusters, split="val_calibration", quantile_levels=levels, min_local_clusters=2
    )

    assert calibrator.effective_support[0] == 3
    assert calibrator.effective_support[1] == 3
    assert calibrator.kappa_val >= 1


def test_metrics_include_coverage_width_and_crps():
    q, y, _, _, levels = _arrays()
    metrics = probability_metrics(q, y, levels)

    assert metrics["coverage_90"] == pytest.approx(1.0)
    assert metrics["width_90"] == pytest.approx(2.0)
    assert metrics["crps"] >= 0
