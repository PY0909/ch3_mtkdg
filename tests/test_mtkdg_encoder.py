"""MTK-DG encoder 测试：输出 shape、有限值、梯度、消融模式、核池化数值。"""

import math

import pytest
import torch

from mtkdg.models.mtkdg import (
    DynamicSensorGraph,
    GatedFusion,
    MTKDGEncoder,
    TimeKernelPooling,
)

B_FD, H_FD, C_FD = 4, 50, 17
B_MT, H_MT, C_MT = 3, 168, 15


def fd_batch(bs=B_FD):
    torch.manual_seed(0)
    x = torch.randn(bs, H_FD, C_FD)
    t = torch.arange(H_FD, dtype=torch.float32).unsqueeze(0).expand(bs, -1).contiguous()
    cond = torch.randn(bs, H_FD, 3)
    return x, t, cond


def mt_batch(bs=B_MT):
    torch.manual_seed(1)
    x = torch.randn(bs, H_MT, C_MT)
    t = torch.arange(H_MT, dtype=torch.float32).unsqueeze(0).expand(bs, -1) * 10.0
    return x, t, None


class TestTimeKernelPooling:
    def test_manual_value(self):
        """核对 h_k = sum(w*x*mask)/(sum(w*mask)+eps) 的手工计算。"""
        torch.manual_seed(2)
        L, C, K = 6, 2, 4
        pool = TimeKernelPooling(n_kernels=K)
        x = torch.randn(1, L, C)
        t = torch.linspace(0.0, 1.0, L).unsqueeze(0)
        mask = torch.ones(1, L, C)
        h, obs = pool(x, t, mask)
        assert h.shape == (1, C, K)
        assert torch.allclose(obs, torch.ones(1, C))

        t_norm = t[0]
        sigma = pool.sigma().detach()
        centers = pool.centers.detach()
        for c in range(C):
            for k in range(K):
                w = torch.exp(-0.5 * ((t_norm - centers[k]) / (sigma[k] + 1e-6)) ** 2)
                expected = (w * x[0, :, c]).sum() / (w.sum() + 1e-6)
                assert torch.isclose(h[0, c, k], expected, atol=1e-5)

    def test_mask_excludes_observations(self):
        pool = TimeKernelPooling(n_kernels=4)
        torch.manual_seed(3)
        x = torch.randn(1, 8, 1)
        t = torch.linspace(0, 1, 8).unsqueeze(0)
        mask = torch.ones(1, 8, 1)
        mask[:, :4, :] = 0.0  # 前半遮蔽
        h_masked, obs = pool(x, t, mask)
        assert torch.allclose(obs, torch.full((1, 1), 0.5))
        # 只用后半观测应不同于全观测
        h_full, _ = pool(x, t, torch.ones_like(mask))
        assert not torch.allclose(h_masked, h_full)
        assert torch.isfinite(h_masked).all()

    def test_all_masked_stays_finite(self):
        pool = TimeKernelPooling(n_kernels=4)
        x = torch.randn(2, 5, 3)
        t = torch.rand(2, 5)
        mask = torch.zeros(2, 5, 3)
        h, obs = pool(x, t, mask)
        assert torch.isfinite(h).all()
        assert torch.allclose(obs, torch.zeros(2, 3))

    def test_constant_time_finite(self):
        """左填充窗口可能出现大量重复时间坐标，必须保持有限。"""
        pool = TimeKernelPooling(n_kernels=4)
        x = torch.randn(2, 50, 17)
        t = torch.zeros(2, 50)
        h, _ = pool(x, t, torch.ones_like(x))
        assert torch.isfinite(h).all()


class TestEncoderShapes:
    def test_fd_shape(self):
        enc = MTKDGEncoder(n_channels=C_FD, d_model=32, condition_dim=3)
        x, t, cond = fd_batch()
        out = enc(x, t, None, cond)
        assert out["z"].shape == (B_FD, C_FD, 32)
        assert out["adjacency"].shape == (B_FD, C_FD, C_FD)
        assert out["gate_weights"].shape == (B_FD, 3)   # 三个尺度
        assert out["fusion_alpha"].shape == (B_FD, C_FD, 32)
        assert out["cond_emb"].shape == (B_FD, 32)

    def test_metropt_shape(self):
        enc = MTKDGEncoder(n_channels=C_MT, d_model=24)
        x, t, _ = mt_batch()
        out = enc(x, t, None, None)
        assert out["z"].shape == (B_MT, C_MT, 24)
        assert out["cond_emb"] is None

    def test_mask_input_supported(self):
        enc = MTKDGEncoder(n_channels=C_FD, d_model=16, condition_dim=3)
        x, t, cond = fd_batch(bs=2)
        mask = torch.ones_like(x)
        mask[:, :10, :] = 0.0
        out = enc(x, t, mask, cond)
        assert torch.isfinite(out["z"]).all()


class TestFiniteValues:
    @pytest.mark.parametrize("graph_mode", ["dynamic", "static", "none"])
    @pytest.mark.parametrize("fusion_mode", ["gated", "add", "self_only", "no_graph"])
    def test_all_ablation_modes_finite(self, graph_mode, fusion_mode):
        enc = MTKDGEncoder(
            n_channels=C_FD, d_model=16, condition_dim=3,
            graph_mode=graph_mode, fusion_mode=fusion_mode, graph_layers=2,
        )
        x, t, cond = fd_batch(bs=2)
        out = enc(x, t, None, cond)
        assert torch.isfinite(out["z"]).all()
        if graph_mode == "none":
            assert out["adjacency"] is None
        else:
            assert torch.isfinite(out["adjacency"]).all()

    def test_large_time_coordinates_finite(self):
        """MetroPT 相对秒坐标 ~1e7 量级也必须稳定。"""
        enc = MTKDGEncoder(n_channels=C_MT, d_model=16)
        x, t, _ = mt_batch(bs=2)
        t = t + 1e7
        out = enc(x, t.contiguous(), None, None)
        assert torch.isfinite(out["z"]).all()

    def test_extreme_values_detected(self):
        enc = MTKDGEncoder(n_channels=4, d_model=8)
        x = torch.randn(2, 10, 4)
        x[0, 0, 0] = float("nan")
        t = torch.arange(10.0).unsqueeze(0).expand(2, -1).contiguous()
        with pytest.raises(RuntimeError, match="non-finite"):
            enc(x, t, None, None)


class TestGradients:
    def test_all_parameters_receive_finite_grads(self):
        enc = MTKDGEncoder(n_channels=C_FD, d_model=24, condition_dim=3, graph_layers=2)
        x, t, cond = fd_batch(bs=4)
        out = enc(x, t, None, cond)
        loss = out["z"].pow(2).mean() + out["adjacency"].pow(2).mean() * 0.0
        loss.backward()
        n_grad, n_missing = 0, []
        for name, p in enc.named_parameters():
            if p.requires_grad:
                n_grad += 1
                if p.grad is None:
                    n_missing.append(name)
                else:
                    assert torch.isfinite(p.grad).all(), name
        assert not n_missing, f"parameters without gradient: {n_missing}"

    def test_kernel_params_get_grads(self):
        enc = MTKDGEncoder(n_channels=5, d_model=8)
        x = torch.randn(2, 20, 5)
        t = torch.rand(2, 20)
        out = enc(x, t, None, None)
        out["z"].sum().backward()
        for i, kern in enumerate(enc.kernels):
            assert kern.centers.grad is not None
            assert kern.log_sigma.grad is not None
            assert torch.isfinite(kern.centers.grad).all()
            assert torch.isfinite(kern.log_sigma.grad).all()


class TestGraphProperties:
    def test_dynamic_adjacency_self_loop_row_structure(self):
        g = DynamicSensorGraph(n_nodes=6, d_model=8, mode="dynamic", self_loop=True)
        z = torch.randn(3, 6, 8)
        adj, _ = g(z)
        eye = torch.eye(6).expand(3, -1, -1)
        assert torch.allclose(adj * eye, eye)          # 对角为 1
        off_diag = adj * (1 - eye)
        assert (off_diag >= 0).all()                   # 非负
        # softmax 后非对角行和 = 1
        assert torch.allclose(off_diag.sum(-1), torch.ones(3, 6), atol=1e-5)

    def test_no_self_loop(self):
        g = DynamicSensorGraph(n_nodes=5, d_model=8, mode="dynamic", self_loop=False)
        adj, _ = g(torch.randn(2, 5, 8))
        assert torch.allclose(adj.sum(-1), torch.ones(2, 5), atol=1e-5)

    def test_none_mode(self):
        g = DynamicSensorGraph(n_nodes=5, d_model=8, mode="none")
        adj, h = g(torch.randn(2, 5, 8))
        assert adj is None
        assert h.shape == (2, 5, 8)

    def test_sample_dependent_adjacency(self):
        """动态图：不同样本应产生不同邻接。"""
        g = DynamicSensorGraph(n_nodes=6, d_model=8, mode="dynamic")
        z = torch.randn(2, 6, 8)
        adj, _ = g(z)
        assert not torch.allclose(adj[0], adj[1], atol=1e-6)

    def test_static_mode_shared_across_samples(self):
        g = DynamicSensorGraph(n_nodes=6, d_model=8, mode="static")
        adj, _ = g(torch.randn(2, 6, 8))
        assert torch.allclose(adj[0], adj[1])


class TestFusion:
    def test_gated_bounds(self):
        f = GatedFusion(d_model=8, condition_dim=0, mode="gated")
        z1, z2 = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        out, alpha = f(z1, z2)
        assert ((alpha >= 0) & (alpha <= 1)).all()

    def test_add_and_self_only(self):
        z1, z2 = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        out, alpha = GatedFusion(8, mode="add")(z1, z2)
        assert torch.allclose(out, z1 + z2)
        assert alpha is None
        out, alpha = GatedFusion(8, mode="self_only")(z1, z2)
        assert torch.allclose(out, z1)

    def test_no_graph_ignores_cross(self):
        out, _ = GatedFusion(8, mode="no_graph")(torch.randn(2, 4, 8), None)
        assert out.shape == (2, 4, 8)
