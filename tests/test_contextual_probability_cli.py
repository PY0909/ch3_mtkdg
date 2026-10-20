import pytest

from scripts.run_contextual_probability import parse_args


def test_cli_requires_checkpoint_protocol_and_unique_run_identity():
    args = parse_args([
        "--dataset", "fd002", "--seed", "2026", "--checkpoint", "point.pt",
        "--mode", "C1", "--output-dir", "runs", "--run-id", "fd002_c1_s2026",
        "--epochs", "80",
    ])
    assert args.training_target == "raw"
    assert args.evaluate_test is False


def test_cli_rejects_missing_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "fd002", "--seed", "2026", "--mode", "C1"])


def test_cpoint_is_raw_label_point_comparator(tmp_path, cmapss_synthetic):
    import numpy as np
    from mtkdg.config import ExperimentConfig
    from mtkdg.data.bundle import bundle_from_cmapss_splits
    from mtkdg.data.cmapss import build_cmapss_splits
    from mtkdg.models.heads import MTKDGPointModel
    from mtkdg.training.checkpoint import save_checkpoint
    from mtkdg.training.probabilistic_runner import ContextualProbabilisticRunner

    train_df, test_df, truth, _ = cmapss_synthetic
    splits = build_cmapss_splits(None, "FD002", training_target="raw", train_df=train_df.copy(),
                                 test_df=test_df.copy(), test_truth=truth)
    bundle = bundle_from_cmapss_splits(splits)
    config = ExperimentConfig.for_dataset("fd002"); config.model.d_model = 8
    point = MTKDGPointModel("rul_regression", 17, 0, config.model,
                            rul_mean=bundle.rul_mean, rul_std=bundle.rul_std)
    path = tmp_path / "point.pt"
    save_checkpoint(str(path), point, extra={"config": config.to_dict()})
    runner = ContextualProbabilisticRunner(bundle, str(path), "fd002", mode="Cpoint", epochs=1)
    history = runner.fit()
    assert np.isfinite(history[0]["train_loss"])
    assert history[0]["loss_name"] == "mse_raw_target"
    assert runner.predict_point(bundle.val).shape == (bundle.val.n_samples,)
    with pytest.raises(RuntimeError, match="no calibrator"):
        runner.calibrate()

    validation = {"predictions": runner.predict_point(bundle.val), "targets": bundle.val.y_raw.reshape(-1)}
    test = runner.evaluate_point_test_once()
    with pytest.raises(RuntimeError, match="only be evaluated once"):
        runner.evaluate_point_test_once()
    run_dir = tmp_path / "cpoint-artifacts"
    runner.save_point_artifacts(str(run_dir), validation, test)

    from mtkdg.evaluation.probability_artifacts import load_probability_artifacts
    manifest, arrays = load_probability_artifacts(str(run_dir))
    assert manifest["mode"] == "Cpoint"
    assert manifest["prediction_type"] == "point"
    assert manifest["evaluated_once"] is True
    assert arrays["test_predictions_raw"].shape == (splits.test.n_samples,)
    assert np.allclose(arrays["test_targets_raw"], splits.test.y_raw.reshape(-1))
