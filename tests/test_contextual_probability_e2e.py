import numpy as np

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.evaluation.probability_artifacts import load_probability_artifacts
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.training.checkpoint import save_checkpoint
from mtkdg.training.probabilistic_runner import ContextualProbabilisticRunner


def test_cpu_end_to_end_saves_reloadable_test_once_artifact(tmp_path, cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    splits = build_cmapss_splits(
        None, "FD002", training_target="raw", train_df=train_df.copy(),
        test_df=test_df.copy(), test_truth=truth,
    )
    bundle = bundle_from_cmapss_splits(splits)
    config = ExperimentConfig.for_dataset("fd002")
    config.model.d_model = 8
    model = MTKDGPointModel(
        "rul_regression", 17, 0, config.model,
        rul_mean=bundle.rul_mean, rul_std=bundle.rul_std,
    )
    checkpoint = tmp_path / "point.pt"
    save_checkpoint(str(checkpoint), model, epoch=2, best_metric=1.0,
                    extra={"config": config.to_dict()})
    runner = ContextualProbabilisticRunner(
        bundle, str(checkpoint), "fd002", mode="C1", epochs=1,
        batch_size=16, device="cpu",
    )
    runner.fit()
    validation = runner.calibrate()
    test = runner.evaluate_test_once()
    run_dir = tmp_path / "artifacts"
    runner.save_artifacts(str(run_dir), validation, test)
    manifest, arrays = load_probability_artifacts(str(run_dir))

    assert manifest["evaluated_once"] is True
    assert manifest["encoder_frozen"] is True
    assert arrays["test_quantiles_raw"].shape == (splits.test.n_samples, 19)
    assert np.allclose(arrays["test_targets_raw"], splits.test.y_raw.reshape(-1))
