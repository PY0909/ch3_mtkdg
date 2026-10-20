"""Baseline encoder 测试：统一合同 [B,H,C]->[B,C,D]，共用任务头，统一协议 smoke。"""

import pytest
import torch

from mtkdg.config import ExperimentConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.models.baselines import BASELINE_NAMES, BaselinePointModel
from mtkdg.training.runner import TrainingRunner, set_seed

B, H, C = 4, 50, 17


def fd_batch(bs=B):
    torch.manual_seed(0)
    x = torch.randn(bs, H, C)
    t = torch.arange(H, dtype=torch.float32).unsqueeze(0).expand(bs, -1).contiguous()
    cond = torch.randn(bs, H, 3)
    return x, t, cond


@pytest.mark.parametrize("name", BASELINE_NAMES)
class TestBaselineContract:
    def test_encoder_output_shape_and_finite(self, name):
        model = BaselinePointModel("rul_regression", name, C, 0, d_model=16, history_len=H)
        x, t, cond = fd_batch()
        z = model.encoder(x, t, None, cond)
        assert z.shape == (B, C, 16)
        assert torch.isfinite(z).all()

    def test_rul_forward_nonnegative(self, name):
        model = BaselinePointModel("rul_regression", name, C, 0, d_model=16, history_len=H,
                                   rul_mean=80.0, rul_std=30.0)
        x, t, cond = fd_batch()
        y = model(x, t, None, cond)
        assert y.shape == (B, 1) and (y >= 0).all()

    def test_state_forward_shape(self, name):
        model = BaselinePointModel("state_forecast", name, 15, 24, d_model=16, history_len=168)
        torch.manual_seed(1)
        x = torch.randn(2, 168, 15)
        t = torch.arange(168, dtype=torch.float32).unsqueeze(0).expand(2, -1).contiguous()
        y = model(x, t, None, None)
        assert y.shape == (2, 24, 15)

    def test_gradients_finite(self, name):
        model = BaselinePointModel("rul_regression", name, C, 0, d_model=16, history_len=H,
                                   rul_mean=80.0, rul_std=30.0)
        x, t, cond = fd_batch()
        model(x, t, None, cond).pow(2).mean().backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, n
                assert torch.isfinite(p.grad).all(), n

    def test_history_len_mismatch_handled(self, name):
        """PatchTST/TimesNet 等对 H 敏感的 encoder 在短窗口下也要能跑。"""
        model = BaselinePointModel("rul_regression", name, C, 0, d_model=8, history_len=H)
        x = torch.randn(2, 20, C)   # H=20 < 构建时的 50
        t = torch.arange(20, dtype=torch.float32).unsqueeze(0).expand(2, -1).contiguous()
        y = model(x, t, None, None)
        assert y.shape == (2, 1)
        assert torch.isfinite(y).all()


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        BaselinePointModel("rul_regression", "mtkdg", C, 0)


@pytest.fixture(scope="module")
def fd_bundle(cmapss_synthetic):
    train_df, test_df, truth, _ = cmapss_synthetic
    s = build_cmapss_splits(
        data_root=None, subset="FD002", history_len=50, stride=2,
        training_cap=125, split_seed=2026,
        train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
    )
    return bundle_from_cmapss_splits(s)


class TestUnifiedProtocolSmoke:
    def test_baseline_one_epoch_through_runner(self, tmp_path, fd_bundle):
        """baseline 必须走与 MTK-DG 相同的 runner/checkpoint/一次性 test 流程。"""
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, lr=1e-3, device="cpu",
            output_dir=str(tmp_path), run_id="gru_smoke", model_name="gru",
        )
        cfg.model.d_model = 12
        set_seed(2026)
        runner = TrainingRunner(cfg, fd_bundle)
        assert runner.model.model_version == "ch3_baselines_v1:gru"
        runner.fit(verbose=False)
        result = runner.run_test(verbose=False)
        assert result["preds_raw"].shape == (fd_bundle.test.n_samples, 1)
        assert (result["preds_raw"] >= 0).all()

    def test_all_baselines_config_accepted(self, fd_bundle, tmp_path):
        for name in BASELINE_NAMES:
            cfg = ExperimentConfig.for_dataset(
                "fd002", epochs=1, batch_size=32, device="cpu",
                output_dir=str(tmp_path), run_id=f"cfg_{name}", model_name=name,
            )
            cfg.model.d_model = 8
            set_seed(0)
            runner = TrainingRunner(cfg, fd_bundle)
            assert runner.model.count_parameters() > 0
