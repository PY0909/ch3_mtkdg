"""artifact schema 测试：run 目录文件齐全，manifest 字段完整。"""

import json
import os

import numpy as np
import pytest

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits, bundle_from_metropt_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.data.metropt import build_metropt_splits
from mtkdg.evaluation.artifacts import save_run_artifacts
from mtkdg.training.runner import TrainingRunner, set_seed

REQUIRED_FILES = [
    "config.json", "manifest.json", "checkpoint_best.pt", "checkpoint_last.pt",
    "history.json", "training_log.csv", "training_curves.png",
    "validation_metrics.json",
]
FD_ONLY = ["test_metrics.json", "predictions_test.npz", "targets_test.npz",
           "per_engine_metrics.csv", "rul_prediction.png"]
MT_ONLY = ["test_metrics.json", "predictions_test.npz", "targets_test.npz",
           "per_sensor_metrics.csv", "state_prediction.png", "per_month_metrics.csv"]

REQUIRED_MANIFEST_KEYS = [
    "run_id", "dataset", "task", "model_version", "seed", "split_identity",
    "feature_schema", "target_schema", "normalizer_identity",
    "checkpoint_identity", "evaluator_version", "sample_counts", "hardware",
    "training_duration_s", "inference_duration_s", "parameter_count",
]


def _tiny_run(bundle, tmp_path, dataset, run_id):
    cfg = ExperimentConfig.for_dataset(
        dataset, epochs=1, batch_size=16, lr=1e-3, device="cpu",
        output_dir=str(tmp_path), run_id=run_id,
    )
    cfg.model.d_model = 8
    set_seed(2026)
    runner = TrainingRunner(cfg, bundle)
    fit_info = runner.fit(verbose=False)
    test_result = runner.run_test(verbose=False)
    save_run_artifacts(runner, bundle, fit_info, test_result)
    return runner, runner.run_dir


@pytest.fixture(scope="module")
def cmapss_bundle(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    s = build_cmapss_splits(
        data_root=None, subset="FD002", history_len=50, stride=2,
        training_cap=125, split_seed=2026,
        train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
    )
    return bundle_from_cmapss_splits(s)


@pytest.fixture(scope="module")
def metropt_bundle(metropt_synthetic):
    s = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
    return bundle_from_metropt_splits(s)


class TestFD002Artifacts:
    @pytest.fixture(scope="class")
    def run(self, cmapss_bundle, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("fd_artifacts")
        return _tiny_run(cmapss_bundle, tmp, "fd002", "artifact_check_fd002")

    def test_all_files_exist(self, run):
        _, run_dir = run
        for name in REQUIRED_FILES + FD_ONLY:
            assert os.path.exists(os.path.join(run_dir, name)), f"missing {name}"

    def test_manifest_fields(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest = json.load(f)
        for key in REQUIRED_MANIFEST_KEYS:
            assert key in manifest, f"manifest missing {key}"
        assert manifest["task"] == "rul_regression"
        assert manifest["checkpoint_identity"]["best_sha256_16"]
        assert manifest["checkpoint_identity"]["best_epoch"] >= 1
        assert manifest["normalizer_identity"]["feature"]
        assert manifest["sample_counts"]["n_test_samples"] > 0
        assert manifest["parameter_count"] > 0
        assert manifest["hardware"]["device"].startswith("cpu")

    def test_npz_schema(self, run, cmapss_bundle):
        _, run_dir = run
        preds = np.load(os.path.join(run_dir, "predictions_test.npz"))["predictions"]
        trues = np.load(os.path.join(run_dir, "targets_test.npz"))["targets"]
        assert preds.shape == (cmapss_bundle.test.n_samples, 1)
        assert trues.shape == (cmapss_bundle.test.n_samples, 1)
        assert (preds >= 0).all()

    def test_test_metrics_json(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "test_metrics.json")) as f:
            payload = json.load(f)
        for key in ("MAE", "RMSE", "NASA"):
            assert key in payload["metrics"]
        assert payload["evaluated_once"] is True

    def test_per_engine_csv(self, run, cmapss_bundle):
        _, run_dir = run
        path = os.path.join(run_dir, "per_engine_metrics.csv")
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert lines[0] == "engine_id,y_true,y_pred,residual,abs_err"
        assert len(lines) - 1 == cmapss_bundle.test.n_samples

    def test_training_log_csv_rows(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "training_log.csv")) as f:
            lines = f.read().strip().split("\n")
        header = lines[0].split(",")
        assert header[:2] == ["epoch", "train_loss"]
        for col in ("val_MAE", "val_RMSE", "val_NASA", "val_RMSE_rawtarget", "lr"):
            assert col in header
        assert len(lines) == 2  # epochs=1

    def test_manifest_git_identity(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert "code_git_sha" in manifest and "code_git_dirty" in manifest
        if manifest["code_git_sha"] is not None:
            assert len(manifest["code_git_sha"]) == 40

    def test_config_pred_len_null_for_rul(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["pred_len"] is None

    def test_val_metrics_rawtarget_diagnostic(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "validation_metrics.json")) as f:
            payload = json.load(f)
        assert payload["best"]["val_RMSE_target"] == "capped(raw cycle)"
        assert payload["best"]["val_RMSE_rawtarget"] >= payload["best"]["val_RMSE"]

    def test_validation_metrics_json(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "validation_metrics.json")) as f:
            payload = json.load(f)
        assert payload["selection_metric"] == "val_RMSE"
        assert payload["best"]["epoch"] >= 1


class TestMetroPTArtifacts:
    @pytest.fixture(scope="class")
    def run(self, metropt_bundle, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("mt_artifacts")
        return _tiny_run(metropt_bundle, tmp, "metropt3", "artifact_check_mt")

    def test_all_files_exist(self, run):
        _, run_dir = run
        for name in REQUIRED_FILES + MT_ONLY:
            assert os.path.exists(os.path.join(run_dir, name)), f"missing {name}"

    def test_manifest_task(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["task"] == "state_forecast"
        assert manifest["split_identity"] == "metropt3_official_first_month_v1"
        assert len(manifest["feature_schema"]["channels"]) == 15

    def test_npz_shape(self, run, metropt_bundle):
        _, run_dir = run
        preds = np.load(os.path.join(run_dir, "predictions_test.npz"))["predictions"]
        trues = np.load(os.path.join(run_dir, "targets_test.npz"))["targets"]
        assert preds.shape == trues.shape
        assert preds.shape[1:] == (8, 15)

    def test_per_sensor_csv(self, run):
        _, run_dir = run
        with open(os.path.join(run_dir, "per_sensor_metrics.csv")) as f:
            lines = f.read().strip().split("\n")
        assert lines[0] == "sensor,channel_index,MAE,RMSE,MAE_norm,RMSE_norm"
        assert len(lines) - 1 == 15

    def test_per_month_metrics(self, run, metropt_bundle):
        _, run_dir = run
        with open(os.path.join(run_dir, "per_month_metrics.csv")) as f:
            lines = f.read().strip().split("\n")
        assert lines[0] == "month,n_windows,overall_MAE,overall_RMSE"
        months = [l.split(",")[0] for l in lines[1:]]
        # 合成数据从 2020-02 起：test 全部 >= 2020-03
        assert all(m >= "2020-03" for m in months)
        with open(os.path.join(run_dir, "test_metrics.json")) as f:
            payload = json.load(f)
        assert set(payload["per_month_metrics"]) == set(months)

    def test_no_test_files_without_evaluate(self, metropt_bundle, tmp_path):
        """不显式 evaluate-test 时不产生 test 文件。"""
        cfg = ExperimentConfig.for_dataset(
            "metropt3", epochs=1, batch_size=16, device="cpu",
            output_dir=str(tmp_path), run_id="no_test", evaluate_test=False,
        )
        cfg.model.d_model = 8
        set_seed(2026)
        runner = TrainingRunner(cfg, metropt_bundle)
        fit_info = runner.fit(verbose=False)
        save_run_artifacts(runner, metropt_bundle, fit_info, None)
        assert not os.path.exists(os.path.join(runner.run_dir, "test_metrics.json"))
        assert not os.path.exists(os.path.join(runner.run_dir, "predictions_test.npz"))
