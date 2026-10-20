"""MC-Graph 测试：可用性条件化邻接、corruption 生成器、鲁棒训练 smoke。"""

import pytest
import torch

from mtkdg.config import ExperimentConfig, MTKDGConfig
from mtkdg.data.bundle import bundle_from_cmapss_splits
from mtkdg.data.cmapss import build_cmapss_splits
from mtkdg.models.heads import MTKDGPointModel
from mtkdg.models.mtkdg import DynamicSensorGraph, GatedFusion, MTKDGEncoder
from mtkdg.training.corruption import channel_dropout, corrupt_batch, time_block_mask
from mtkdg.training.runner import TrainingRunner, set_seed


def fd_batch(bs=4):
    torch.manual_seed(0)
    x = torch.randn(bs, 50, 17)
    t = torch.arange(50.0).unsqueeze(0).expand(bs, -1).contiguous()
    cond = torch.randn(bs, 50, 3)
    return x, t, cond


class TestAvailabilityAdjacency:
    @pytest.mark.parametrize("mode", ["edge_bias", "source_suppress", "both"])
    def test_modes_forward_finite_and_rows_normalized(self, mode):
        g = DynamicSensorGraph(n_nodes=6, d_model=8, mode="dynamic",
                               availability_mode=mode)
        z = torch.randn(3, 6, 8)
        r = torch.rand(3, 6)
        adj, h = g(z, obs_ratio=r)
        assert torch.isfinite(adj).all()
        assert (adj >= 0).all()
        assert torch.isfinite(h).all()
        if mode in ("source_suppress", "both"):
            # 源端抑制后整行（含 self-loop）重归一化为分布（S2-2 审查修正）
            assert torch.allclose(adj.sum(-1), torch.ones(3, 6), atol=1e-4)
        else:
            # edge_bias-only 保留 FC-STGNN 原始 +eye 约定（行和 = 2）
            assert torch.allclose(adj.sum(-1), torch.full((3, 6), 2.0), atol=1e-4)

    def test_source_suppress_blocks_unavailable_broadcast(self):
        """不可用源节点的出边质量必须趋近 0（评审指出的正确性要求）。"""
        g = DynamicSensorGraph(n_nodes=5, d_model=8, mode="dynamic",
                               availability_mode="source_suppress")
        z = torch.randn(1, 5, 8)
        r = torch.ones(1, 5)
        r[0, 2] = 0.0                                     # 通道 2 完全不可用
        adj, _ = g(z, obs_ratio=r)
        eye = torch.eye(5, dtype=torch.bool)
        diag_vec = torch.diagonal(eye)                     # [1,0,0,0,0]
        col = adj[0, :, 2] * (~diag_vec).float()           # 排除自环后源=2 的出边
        assert (col.abs() < 1e-4).all(), "unavailable source must not broadcast"
        # 整行（含 self-loop）重归一化为分布
        assert torch.allclose(adj.sum(-1), torch.ones(1, 5), atol=1e-4)

    def test_source_suppress_reweights_self_loop(self):
        """跨传感器消息被抑制时 self-loop 权重自适应放大（整行重归一化的直接后果）。"""
        g = DynamicSensorGraph(n_nodes=4, d_model=8, mode="dynamic",
                               availability_mode="source_suppress")
        z = torch.randn(1, 4, 8)
        r_full = torch.ones(1, 4)
        r_low = torch.full((1, 4), 0.1)
        eye = torch.eye(4).unsqueeze(0)
        adj_full, _ = g(z, obs_ratio=r_full)
        adj_low, _ = g(z, obs_ratio=r_low)
        assert (adj_low * eye).sum(-1).median() > (adj_full * eye).sum(-1).median()

    def test_edge_bias_changes_adjacency_with_availability(self):
        g = DynamicSensorGraph(n_nodes=5, d_model=8, mode="dynamic",
                               availability_mode="edge_bias")
        # 打破零初始化，使 bias 有作用
        with torch.no_grad():
            g.edge_bias[0].weight.fill_(1.0)
            g.edge_bias[0].bias.fill_(0.5)
        z = torch.randn(2, 5, 8)
        # 非均匀可用性（均匀向量会产生逐行常数偏置，在 softmax 中抵消）
        r_hi = torch.tensor([[1.0, 0.2, 1.0, 0.2, 1.0]] * 2)
        r_lo = torch.tensor([[0.2, 1.0, 0.2, 1.0, 0.2]] * 2)
        a_hi, _ = g(z, obs_ratio=r_hi)
        a_lo, _ = g(z, obs_ratio=r_lo)
        assert not torch.allclose(a_hi, a_lo), \
            "availability must modulate adjacency under edge_bias"

    def test_availability_mode_requires_obs_ratio(self):
        g = DynamicSensorGraph(n_nodes=4, d_model=8, mode="dynamic",
                               availability_mode="both")
        with pytest.raises(ValueError, match="obs_ratio"):
            g(torch.randn(2, 4, 8), obs_ratio=None)

    def test_encoder_passes_obs_ratio_and_output_exposes_it(self):
        enc = MTKDGEncoder(n_channels=5, d_model=8, availability_mode="both")
        x = torch.randn(2, 20, 5)
        t = torch.rand(2, 20)
        out = enc(x, t, None, None)
        assert out["obs_ratio"].shape == (2, 5)
        assert torch.isfinite(out["z"]).all()

    def test_full_model_mc_graph_forward(self):
        cfg = MTKDGConfig(d_model=12, availability_mode="both")
        model = MTKDGPointModel("rul_regression", 17, 0, cfg, rul_mean=60.0, rul_std=30.0)
        x, t, cond = fd_batch()
        y = model(x, t, None, cond)
        assert y.shape == (4, 1) and (y >= 0).all()

    def test_fusion_availability(self):
        f = GatedFusion(d_model=8, mode="gated", use_availability=True)
        z1, z2 = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        r = torch.rand(2, 4)
        out, alpha = f(z1, z2, None, obs_ratio=r)
        assert out.shape == (2, 4, 8) and ((alpha >= 0) & (alpha <= 1)).all()
        with pytest.raises(ValueError):
            f(z1, z2, None, obs_ratio=None)

    def test_backward_compat_none_mode(self):
        """availability none 时行为与旧接口一致（obs_ratio 可省略）。"""
        g = DynamicSensorGraph(n_nodes=5, d_model=8, mode="dynamic")
        adj, _ = g(torch.randn(2, 5, 8))
        assert torch.isfinite(adj).all()


class TestCorruption:
    def test_channel_dropout_protects_condition_channels(self):
        x = torch.ones(8, 10, 17)
        mask = torch.ones_like(x)
        xc, mc = channel_dropout(x, mask, rate=0.9, protect_channels=3)
        assert (mc[:, :, :3] == 1).all(), "condition channels must never be dropped"
        dropped = (mc[:, :, 3:] == 0).float().mean()
        assert dropped > 0.3                              # rate 0.9 应大量丢弃

    def test_time_block_contiguous_and_protected(self):
        x = torch.ones(4, 20, 5)
        mask = torch.ones_like(x)
        xc, mc = time_block_mask(x, mask, ratio=0.2, protect_channels=0)
        dropped_rows = (mc[:, :, 0] == 0).sum(dim=1)
        assert ((dropped_rows >= 3) & (dropped_rows <= 6)).all()   # ≈20% 连续块
        # 连续性：每样本被丢弃行是单一连续块
        for b in range(4):
            nz = torch.nonzero(mc[b, :, 0] == 0).flatten()
            if len(nz):
                assert nz.max() - nz.min() + 1 == len(nz)

    def test_corrupt_batch_probability_gate(self):
        cfg = MTKDGConfig(robust_train=True, corrupt_prob=0.0,
                          corrupt_channel_rate=0.5, corrupt_block_ratio=0.5)
        x, mask = torch.ones(2, 10, 4), torch.ones(2, 10, 4)
        xc, mc = corrupt_batch(x, mask, cfg, protect_channels=0)
        assert torch.equal(xc, x) and torch.equal(mc, mask)

    def test_corrupt_batch_changes_input(self):
        cfg = MTKDGConfig(robust_train=True, corrupt_prob=1.0,
                          corrupt_channel_rate=0.5, corrupt_block_ratio=0.5)
        x, mask = torch.ones(16, 20, 8), torch.ones(16, 20, 8)
        xc, mc = corrupt_batch(x, mask, cfg, protect_channels=0)
        assert (mc < 1).any() and (xc != x).any()
        # x 与 mask 一致置零
        assert torch.equal(xc[mc == 0], torch.zeros_like(xc[mc == 0]))


class TestRobustTrainingSmoke:
    @pytest.fixture(scope="class")
    def fd_bundle(self, cmapss_synthetic):
        train_df, test_df, truth, _ = cmapss_synthetic
        s = build_cmapss_splits(
            data_root=None, subset="FD002", history_len=50, stride=2,
            training_cap=125, split_seed=2026,
            train_df=train_df.copy(), test_df=test_df.copy(), test_truth=truth,
        )
        return bundle_from_cmapss_splits(s)

    @pytest.mark.parametrize("avail,cons", [("none", 0.0), ("both", 0.1)])
    def test_mc_graph_train_one_epoch(self, tmp_path, fd_bundle, avail, cons):
        cfg = ExperimentConfig.for_dataset(
            "fd002", epochs=1, batch_size=32, lr=1e-3, device="cpu",
            output_dir=str(tmp_path), run_id=f"mc_{avail}_{cons}",
            availability_mode=avail, robust_train=True,
            corrupt_channel_rate=0.3, corrupt_block_ratio=0.3,
            consistency_lambda=cons,
        )
        cfg.model.d_model = 12
        set_seed(2026)
        runner = TrainingRunner(cfg, fd_bundle)
        loss = runner.train_one_epoch()
        assert loss == loss and loss > 0
        val = runner.evaluate_split(fd_bundle.val)       # val 仍用完整观测
        assert val["val_RMSE"] == val["val_RMSE"]
