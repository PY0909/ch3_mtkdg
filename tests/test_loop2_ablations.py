"""Loop-2 预注册消融测试：V1 分工况归一化、V2 raw 头（单/双）、V3 退化通道剔除、V4 delta 头。"""

import numpy as np
import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits
from mtkdg.data.cmapss import build_cmapss_splits, condition_codes
from mtkdg.data.normalizer import PerConditionScaler, TrainOnlyStandardScaler
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.training.runner import TrainingRunner, set_seed


def _fd_bundle(cmapss_synthetic, norm_mode="global"):
    train_df, test_df, truth, _ = cmapss_synthetic
    s = build_cmapss_splits(
        data_root=None, subset="FD002", history_len=50, stride=2,
        training_cap=125, split_seed=2026, norm_mode=norm_mode,
        train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
    )
    return s, bundle_from_cmapss_splits(s)


class TestV1PerConditionNorm:
    def test_condition_codes_deterministic(self, cmapss_synthetic):
        train_df = cmapss_synthetic[0]
        codes = condition_codes(train_df[["setting1", "setting2", "setting3"]].to_numpy())
        assert codes.min() == 0 and len(np.unique(codes)) >= 3   # 合成 settings 多工况
        assert (condition_codes(train_df[["setting1", "setting2", "setting3"]].to_numpy()) == codes).all()

    def test_per_condition_fit_train_only(self, cmapss_synthetic):
        train_df = cmapss_synthetic[0]
        s_global, b_global = _fd_bundle(cmapss_synthetic, "global")
        s_pc, b_pc = _fd_bundle(cmapss_synthetic, "per_condition")
        # 两种模式产生不同的标准化结果与不同的身份指纹
        assert not np.allclose(b_global.train.x, b_pc.train.x)
        assert b_pc.feature_scaler.identity() != b_global.feature_scaler.identity()
        assert b_pc.meta["norm_mode"] == "per_condition"
        # 全部有限（settings 在工况内近常数由 min_std 保护）
        for split in (b_pc.train, b_pc.val, b_pc.test):
            assert np.isfinite(split.x).all()
            assert np.abs(split.x).max() < 1e3

    def test_unseen_condition_falls_back(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, (200, 4))
        codes = np.zeros(200, dtype=np.int64); codes[100:] = 1
        scaler = PerConditionScaler(min_std=0.01).fit(x, codes)
        # code 2 未见：回退全局 scaler，仍有限
        out = scaler.transform(rng.normal(0, 1, (10, 4)), np.full(10, 2))
        assert np.isfinite(out).all()

    def test_per_condition_isolates_condition_variance(self, cmapss_synthetic):
        """同一样本在 per-condition 下应消除工况均值偏移（粗检验：全局方差更大）。"""
        _, b_global = _fd_bundle(cmapss_synthetic, "global")
        _, b_pc = _fd_bundle(cmapss_synthetic, "per_condition")
        assert b_pc.train.x.std() <= b_global.train.x.std() + 1e-6


class TestV2RawHeads:
    @pytest.fixture(scope="class")
    def fd(self, cmapss_synthetic):
        return _fd_bundle(cmapss_synthetic)

    def test_raw_single_forward_and_loss(self, tmp_path, fd):
        s, bundle = fd
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, device="cpu",
            output_dir=str(tmp_path), run_id="raw_single", rul_head="raw_single",
            checkpoint_metric="val_RMSE_rawtarget",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, bundle)
        assert runner.model.rul_head_mode == "raw_single"
        loss = runner.train_one_epoch()
        assert np.isfinite(loss)
        # test-time 输出来自 raw 头（对 raw 目标评估）
        preds, trues = runner.predict_split(bundle.test)
        assert (preds >= 0).all()

    def test_dual_head_combined_loss(self, tmp_path, fd):
        s, bundle = fd
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, device="cpu",
            output_dir=str(tmp_path), run_id="dual", rul_head="dual",
            checkpoint_metric="val_RMSE_rawtarget",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, bundle)
        model = runner.model
        assert hasattr(model, "head") and hasattr(model, "aux_head")  # capped + raw 双头
        batch = next(runner._iter_batches(bundle.train, shuffle=False))
        heads = model.forward_train(batch["x"], batch["t"], batch["mask"], batch["condition"])
        assert set(heads) == {"capped", "raw"}
        loss = runner._compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        # 两个头都收到梯度
        assert model.head.mlp[-1].weight.grad is not None
        assert model.aux_head.mlp[-1].weight.grad is not None

    def test_dual_e2e_and_selection_metric(self, tmp_path, fd, capsys):
        s, bundle = fd
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, device="cpu",
            output_dir=str(tmp_path), run_id="dual_e2e", rul_head="dual",
            checkpoint_metric="val_RMSE_rawtarget",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, bundle)
        runner.fit(verbose=False)
        # 选型指标为 raw-target（best_metric 应等于 best epoch 的 val_RMSE_rawtarget）
        idx = runner.history["epoch"].index(runner.best_epoch)
        assert runner.best_metric == pytest.approx(
            runner.history["val_RMSE_rawtarget"][idx], rel=1e-6)
        result = runner.run_test(verbose=False)
        assert (result["preds_raw"] >= 0).all()

    def test_single_mode_unchanged(self, tmp_path, fd):
        """V0 单头模式行为不变（回归保护）。"""
        s, bundle = fd
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, device="cpu",
            output_dir=str(tmp_path), run_id="v0",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, bundle)
        model = runner.model
        assert not hasattr(model, "aux_head")
        heads = model.forward_train(*[torch.randn(2, 50, 17), torch.arange(50.).expand(2, -1).contiguous(),
                                      None, torch.randn(2, 50, 3)])
        assert set(heads) == {"capped"}


class TestV3V4Metropt:
    def test_exclude_degenerate(self, metropt_synthetic):
        from mtkdg.data.bundle import _exclude_degenerate_channels, bundle_from_metropt_splits
        from mtkdg.data.metropt import build_metropt_splits

        s = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
        bundle = bundle_from_metropt_splits(s)
        stds = np.asarray(bundle.feature_scaler.std)
        expected_drop = [i for i, v in enumerate(stds) if v < 0.05]
        bundle2 = _exclude_degenerate_channels(bundle)
        if expected_drop:
            assert (bundle2.train.x[:, :, expected_drop] == 0).all()
            assert (bundle2.train.mask[:, :, expected_drop] == 0).all()
            names = bundle2.feature_schema["channels"]
            assert bundle2.meta["excluded_degenerate_channels"] == [names[i] for i in expected_drop]
            # 输出头合同不变：y 仍含全部通道
            assert bundle2.train.y.shape[-1] == len(names)

    def test_delta_head_persistence_skip(self):
        cfg = ExperimentConfig.for_dataset("metropt3")
        cfg.model.state_head_mode = "delta"
        cfg.model.d_model = 8
        model = MTKDGPointModel("state_forecast", 15, 8, cfg.model)
        model.eval()   # 关闭 dropout，保证 forward 与 forward_train 一致
        x = torch.randn(3, 24, 15)
        t = torch.arange(24.0).unsqueeze(0).expand(3, -1).contiguous()
        with torch.no_grad():
            out = model(x, t, None, None)
            heads = model.forward_train(x, t, None, None)
        assert out.shape == (3, 8, 15)
        assert torch.isfinite(out).all()
        assert torch.allclose(heads["state"], out)

    def test_delta_e2e_smoke(self, tmp_path, metropt_synthetic):
        from mtkdg.data.bundle import bundle_from_metropt_splits
        from mtkdg.data.metropt import build_metropt_splits

        s = build_metropt_splits(metropt_synthetic, history_len=24, pred_len=8, stride=6)
        bundle = bundle_from_metropt_splits(s)
        cfg = ExperimentConfig.for_dataset(
            "metropt3", epochs=1, batch_size=32, device="cpu",
            output_dir=str(tmp_path), run_id="delta", state_head_mode="delta",
        )
        cfg.model.d_model = 8
        set_seed(0)
        runner = TrainingRunner(cfg, bundle)
        runner.fit(verbose=False)
        result = runner.run_test(verbose=False)
        assert np.isfinite(result["preds_raw"]).all()
