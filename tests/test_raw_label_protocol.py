import numpy as np

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import build_bundle
from mtkdg.data.cmapss import build_cmapss_splits


def _build(train_df, test_df, truth, training_target):
    return build_cmapss_splits(
        data_root=None,
        subset="FD002",
        history_len=50,
        stride=1,
        training_cap=125,
        split_seed=2026,
        train_df=train_df.copy(),
        test_df=test_df.copy(),
        test_truth=truth,
        training_target=training_target,
    )


def test_raw_training_target_preserves_complete_train_and_validation_rul(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    splits = _build(train_df, test_df, truth, "raw")

    assert (splits.train.y_raw > 125).any()
    assert np.allclose(splits.val.y_raw, splits.val.y_raw_uncapped)
    assert np.allclose(splits.train.y, splits.target_scaler.transform(splits.train.y_raw))
    assert np.allclose(splits.val.y, splits.target_scaler.transform(splits.val.y_raw))
    assert np.allclose(np.sort(splits.test.y_raw.reshape(-1)), np.sort(truth))
    assert splits.target_schema["training_target"] == "raw"
    assert splits.meta["training_target"] == "raw"


def test_raw_target_scaler_fits_train_engines_only(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    splits = _build(train_df, test_df, truth, "raw")
    train_values = splits.train.y_raw.astype(np.float64)

    assert np.isclose(splits.target_scaler.mean[0], train_values.mean())
    assert np.isclose(splits.target_scaler.std[0], train_values.std())
    assert splits.meta["train_label_sha256"]
    assert splits.meta["val_label_sha256"]


def test_raw_and_capped_protocols_have_distinct_identity(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    raw = _build(train_df, test_df, truth, "raw")
    capped = _build(train_df, test_df, truth, "capped")

    assert raw.identity != capped.identity
    assert (capped.train.y_raw <= 125).all()
    assert capped.target_schema["training_target"] == "capped"


def test_config_default_preserves_capped_protocol():
    config = ExperimentConfig.for_dataset("fd002")

    assert config.training_target == "capped"


def test_bundle_passes_explicit_raw_target(cmapss_synthetic, monkeypatch):
    train_df, test_df, truth, _ = cmapss_synthetic
    from mtkdg.data import bundle as bundle_module

    monkeypatch.setattr(bundle_module, "build_cmapss_splits", lambda **kwargs: _build(
        train_df, test_df, truth, kwargs["training_target"]
    ))
    config = ExperimentConfig.for_dataset("fd002", training_target="raw")
    bundle = build_bundle(config)

    assert bundle.meta["training_target"] == "raw"
