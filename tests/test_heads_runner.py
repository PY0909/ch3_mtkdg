"""任务头 + 完整模型 + training runner 测试（合成数据，CPU）。"""

import os

import numpy as np
import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits, bundle_from_metropt_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.data.metropt import build_metropt_splits
from mtkdg.models.heads import MTKDGPointModel, RULHead, StateHead
from mtkdg.training.checkpoint import load_checkpoint, save_checkpoint
from mtkdg.training.runner import TrainingRunner, set_seed


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


class TestHeads:
    def test_state_head_shape(self):
        head = StateHead(d_model=16, n_channels=15, pred_len=24)
        z = torch.randn(4, 15, 16)
        y = head(z)
        assert y.shape == (4, 24, 15)
        assert torch.isfinite(y).all()

    def test_rul_head_nonnegative_and_conditionless(self):
        head = RULHead(d_model=16, rul_mean=80.0, rul_std=30.0)
        z = torch.randn(4, 17, 16)
        y = head(z, None)                      # 无工况条件（A6 消融路径）
        assert y.shape == (4, 1)
        assert (y >= 0).all()
        # bias 初始化应使初始输出接近 train 平均 RUL
        assert 30.0 < y.mean() < 150.0

    def test_rul_head_with_condition(self):
        head = RULHead(d_model=16, rul_mean=80.0, rul_std=30.0)
        z = torch.randn(4, 17, 16)
        cond = torch.randn(4, 16)
        y = head(z, cond)
        assert y.shape == (4, 1) and (y >= 0).all()


class TestFullModel:
    def test_rul_model_forward(self, cmapss_bundle):
        cfg = ExperimentConfig.for_dataset("fd002")
        cfg.model.d_model = 16
        model = MTKDGPointModel("rul_regression", 17, 0, cfg.model,
                                rul_mean=cmapss_bundle.rul_mean,
                                rul_std=cmapss_bundle.rul_std)
        split = cmapss_bundle.train
        out = model(torch.from_numpy(split.x[:4]), torch.from_numpy(split.t[:4]),
                    torch.from_numpy(split.mask[:4]), torch.from_numpy(split.condition[:4]))
        assert out.shape == (4, 1)
        assert (out >= 0).all()
        assert torch.isfinite(out).all()

    def test_state_model_forward(self, metropt_bundle):
        cfg = ExperimentConfig.for_dataset("metropt3")
        cfg.model.d_model = 16
        model = MTKDGPointModel("state_forecast", 15, 8, cfg.model)
        split = metropt_bundle.train
        out = model(torch.from_numpy(split.x[:4]), torch.from_numpy(split.t[:4]),
                    torch.from_numpy(split.mask[:4]), None)
        assert out.shape == (4, 8, 15)
        assert torch.isfinite(out).all()


class TestCheckpoint:
    def test_save_restore_roundtrip(self, tmp_path, cmapss_bundle):
        cfg = ExperimentConfig.for_dataset("fd002")
        cfg.model.d_model = 16
        set_seed(0)
        model = MTKDGPointModel("rul_regression", 17, 0, cfg.model,
                                rul_mean=60.0, rul_std=30.0)
        x = torch.randn(3, 50, 17)
        t = torch.arange(50.0).unsqueeze(0).expand(3, -1).contiguous()
        cond = torch.randn(3, 50, 3)
        before = model(x, t, None, cond)

        path = str(tmp_path / "checkpoint_best.pt")
        save_checkpoint(path, model, epoch=3, best_metric=12.5,
                        extra={"config": cfg.to_dict()})
        payload = load_checkpoint(path)
        assert payload["epoch"] == 3
        assert payload["best_metric"] == 12.5

        # 修改权重后恢复，应还原预测
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        model2 = MTKDGPointModel("rul_regression", 17, 0, cfg.model,
                                 rul_mean=60.0, rul_std=30.0)
        load_checkpoint(path, model=model2)
        after = model2(x, t, None, cond)
        assert torch.allclose(before, after, atol=1e-6)


class TestRunnerSmoke:
    def test_cmapss_one_batch_training_smoke(self, tmp_path, capsys, cmapss_bundle):
        """CPU smoke：小模型 2 epochs，验证日志、checkpoint、一次性 test。"""
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=2, batch_size=16, lr=1e-3, patience=2,
            device="cpu", output_dir=str(tmp_path), run_id="smoke",
        )
        cfg.model.d_model = 12
        set_seed(2026)
        runner = TrainingRunner(cfg, cmapss_bundle)
        fit_info = runner.fit(verbose=True)

        captured = capsys.readouterr().out
        assert "Epoch 001/002" in captured
        assert "train_loss=" in captured
        assert "val_RMSE=" in captured
        assert "Best checkpoint: epoch=" in captured
        assert fit_info["best_epoch"] >= 1

        # best / last checkpoint 都已保存
        assert (runner.run_dir + "/checkpoint_best.pt").startswith(str(tmp_path))
        import os
        assert os.path.exists(os.path.join(runner.run_dir, "checkpoint_best.pt"))
        assert os.path.exists(os.path.join(runner.run_dir, "checkpoint_last.pt"))

        # test 只能评估一次
        result = runner.run_test(verbose=True)
        assert result["preds_raw"].shape == (cmapss_bundle.test.n_samples, 1)
        assert (result["preds_raw"] >= 0).all()
        out = capsys.readouterr().out
        assert "Test | MAE=" in out
        with pytest.raises(RuntimeError):
            runner.run_test()

    def test_metropt_training_smoke(self, tmp_path, capsys, metropt_bundle):
        cfg = ExperimentConfig.for_dataset(
            "metropt3", epochs=1, batch_size=32, lr=1e-3, patience=1,
            device="cpu", output_dir=str(tmp_path), run_id="smoke_mt",
        )
        cfg.model.d_model = 12
        set_seed(2026)
        runner = TrainingRunner(cfg, metropt_bundle)
        runner.fit(verbose=True)
        out = capsys.readouterr().out
        assert "val_overall_MAE=" in out and "val_overall_RMSE=" in out
        result = runner.run_test(verbose=True)
        assert result["preds_raw"].shape == (metropt_bundle.test.n_samples, 8, 15)
        # 预测已反归一化到 raw unit：与真值均值同一量级（而非标准化空间的 ~0）
        pred_scale = float(np.abs(result["preds_raw"]).mean())
        true_scale = float(np.abs(result["trues_raw"]).mean())
        assert 0.05 * true_scale < pred_scale < 20 * true_scale

    def test_loss_decreases_on_tiny_setup(self, tmp_path, cmapss_bundle):
        """合成退化信号可学习：6 epochs 内 train loss 应明显下降。"""
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=6, batch_size=64, lr=1e-2, patience=6,
            device="cpu", output_dir=str(tmp_path), run_id="lr_check",
        )
        cfg.model.d_model = 16
        set_seed(0)
        runner = TrainingRunner(cfg, cmapss_bundle)
        losses = [runner.train_one_epoch() for _ in range(6)]
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
        assert min(losses[1:]) < losses[0]


class TestCudaSmoke:
    def test_cuda_one_epoch_if_available(self, tmp_path, capsys, cmapss_bundle):
        """CUDA 路径 smoke：设备可用时跑 1 epoch（非正式实验，仅验证代码路径）。"""
        import torch as th

        if not th.cuda.is_available():
            pytest.skip("CUDA not available")
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=16, lr=1e-3, patience=1,
            device="cuda", output_dir=str(tmp_path), run_id="cuda_smoke",
        )
        cfg.model.d_model = 12
        set_seed(2026)
        runner = TrainingRunner(cfg, cmapss_bundle)
        assert runner.device.type == "cuda"
        assert next(runner.model.parameters()).device.type == "cuda"
        runner.fit(verbose=True)
        out = capsys.readouterr().out
        assert "Epoch 001/001" in out
        result = runner.run_test(verbose=True)
        assert np.isfinite(result["preds_raw"]).all()
        # checkpoint 从 GPU 保存后可在 CPU 恢复
        model_cpu = MTKDGPointModel("rul_regression", 17, 0, cfg.model,
                                    rul_mean=60.0, rul_std=30.0)
        load_checkpoint(
            os.path.join(runner.run_dir, "checkpoint_best.pt"),
            model=model_cpu, map_location="cpu",
        )
