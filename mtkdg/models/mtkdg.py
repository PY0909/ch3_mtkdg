"""MTK-DG Encoder（Multi-Scale Time-Kernel Dynamic Sensor-Graph Encoder）。

结构链路（第三章主干，只做点预测表示）：

```
x [B,H,C], t [B,H], mask [B,H,C], condition [B,H,3]?
  -> 三个历史尺度（最近 1/4、1/2、完整窗口）
  -> 每尺度 KAF 风格 K-Gaussian 时间核池化（每通道独立）
  -> ScaleGate 轻量门控融合 + 传感器身份编码
  -> 动态传感器图 A = softmax(phi(Z) phi(Z)^T / sqrt(d))，G = A Z W
  -> self-history 与 cross-sensor 门控融合 z_out = a*z_self + (1-a)*z_cross
  -> z [B, C, d]（供任务头使用）
```

所有数值路径带有限值检查；图模式支持 dynamic / static / none（消融 A3/A4）；
融合模式支持 gated / add / self_only / no_graph（消融 A5/A7）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "mtkdg_encoder_v1"
DEFAULT_SCALES = (0.25, 0.5, 1.0)
EPS = 1e-6


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite values detected in {name}")


class TimeKernelPooling(nn.Module):
    """KAF 风格 K-Gaussian 时间核池化（逐通道）。

    w_k(t) = exp(-0.5 * ((t - c_k) / sigma_k)^2)
    h_k    = sum_t w_k(t) * m_t * x_t / (sum_t w_k(t) * m_t + eps)

    输入时间坐标在窗口内 min-max 归一化到 [0,1]；核中心/宽度可学习。
    输出 [B, C, K]，外加每通道有效观测比例 [B, C, 1]。
    """

    def __init__(self, n_kernels: int = 4, init_sigma: float = 0.3):
        super().__init__()
        self.n_kernels = n_kernels
        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, n_kernels))
        self.log_sigma = nn.Parameter(torch.full((n_kernels,), math.log(init_sigma)))
        self.init_sigma = init_sigma

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp() + 1e-6

    def forward(
        self,
        x: torch.Tensor,      # [B, L, C]
        t: torch.Tensor,      # [B, L]
        mask: torch.Tensor,   # [B, L, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        # 窗口内时间归一化（全部相同坐标时保持 0，保证有限）
        t_min = t.min(dim=1, keepdim=True).values
        t_max = t.max(dim=1, keepdim=True).values
        t_norm = (t - t_min) / (t_max - t_min + EPS)          # [B, L]

        centers = self.centers.view(1, 1, -1)                  # [1,1,K]
        sigma = self.sigma().view(1, 1, -1)                    # [1,1,K]
        z = (t_norm.unsqueeze(-1) - centers) / (sigma + EPS)   # [B,L,K]
        w = torch.exp(-0.5 * z * z)                            # [B,L,K]

        m = mask.unsqueeze(-1)                                 # [B,L,C,1]
        w_m = w.unsqueeze(2) * m                               # [B,L,C,K]
        num = (w_m * x.unsqueeze(-1)).sum(dim=1)               # [B,C,K]
        den = w_m.sum(dim=1) + EPS                             # [B,C,K]
        h = num / den                                          # [B,C,K]

        # 全遮蔽通道：分子分母同时为 0，h 仍有限（eps 保护），观测比例暴露给下游
        obs_ratio = mask.mean(dim=1, keepdim=False)            # [B,C]
        _check_finite(h, "time_kernel_pooling.h")
        return h, obs_ratio


class ScaleGate(nn.Module):
    """DisMS-TS 风格轻量尺度门控：从各尺度摘要产生 softmax 权重 [B, S]。"""

    def __init__(self, n_channels: int, n_scales: int, hidden_dim: int = 64):
        super().__init__()
        summary_dim = n_channels * 2 + n_scales * 2
        self.net = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_scales),
        )

    def forward(self, scale_summaries: torch.Tensor, x_stats: torch.Tensor) -> torch.Tensor:
        """scale_summaries [B,S,stats]，x_stats [B,C*2] -> [B,S]。"""
        s_mean = scale_summaries.mean(dim=-1)                  # [B,S]
        s_std = scale_summaries.std(dim=-1)                    # [B,S]
        summary = torch.cat([x_stats, s_mean, s_std], dim=-1)
        return torch.softmax(self.net(summary), dim=-1)


class DynamicSensorGraph(nn.Module):
    """传感器级动态图（FC-STGNN 风格 dot-product 图）。

    - dynamic：A = softmax(phi(Z) phi(Z)^T / sqrt(d))，可加 self-loop；
    - static  ：可学习静态邻接（softmax 后加 self-loop），作为 A3 消融；
    - none    ：不构图（A4 消融），forward 返回 None。

    每层：Z_{l+1} = Z_l + A_l (Z_l W_l)（残差，限制数值范围）。

    MC-Graph（第四章机制创新）：availability_mode 控制传感器可用性
    如何调制消息传递（obs_ratio = 时间核池化输出的逐通道观测比例）：
    - "edge_bias"      ：logits += w·[r_i, r_j, r_i*r_j]（目的端/源端可靠性偏置）；
    - "source_suppress"：softmax 后按源端可用性抑制列并逐行重归一化
      （不可用源节点不能广播——评审指出的实现要点，有限 logit 相乘不够）；
    - "both"           ：两者同时使用；
    - "none"           ：原始图（消融对照）。
    """

    AVAILABILITY_MODES = ("none", "edge_bias", "source_suppress", "both")

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        mode: str = "dynamic",
        n_layers: int = 1,
        self_loop: bool = True,
        availability_mode: str = "none",
    ):
        super().__init__()
        if mode not in ("dynamic", "static", "none"):
            raise ValueError(f"unknown graph mode: {mode}")
        if availability_mode not in self.AVAILABILITY_MODES:
            raise ValueError(f"unknown availability_mode: {availability_mode}")
        self.mode = mode
        self.availability_mode = availability_mode
        self.n_layers = n_layers if mode != "none" else 0
        self.self_loop = self_loop
        self.n_nodes = n_nodes

        if self.n_layers > 0:
            self.phi = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.n_layers)])
            self.w = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.n_layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_layers)])
            if mode == "static":
                self.static_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
                with torch.no_grad():
                    self.static_logits.fill_diagonal_(0.0)
            if availability_mode in ("edge_bias", "both"):
                # b(r_i, r_j) 作用于每条边：输入 [r_i, r_j, r_i*r_j]
                self.edge_bias = nn.ModuleList(
                    [nn.Linear(3, 1) for _ in range(self.n_layers)]
                )
                for eb in self.edge_bias:
                    nn.init.zeros_(eb.weight)
                    nn.init.zeros_(eb.bias)

    @staticmethod
    def _suppress_unavailable_sources(adj: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """softmax 后按源端可用性抑制非对角列，self-loop 不受抑制，整行重归一化。

        A_ij = 从源 j 到目的 i 的消息权重（msg = A @ Z）。
        不可用源（r_j≈0）的出边质量被移除；self-loop 保留（设备自身观测永远
        可为自己背书），整行重归一化后每行仍是分布，self-loop 权重自适应
        放大以补偿被抑制的跨传感器消息（S2-2 审查确认的语义）。
        """
        N = adj.size(-1)
        eye = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0)
        off = adj * (~eye).to(adj.dtype) * r.unsqueeze(-2)    # 按源（列）抑制
        sup = off + adj * eye.to(adj.dtype)
        return sup / sup.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _adjacency(self, z: torch.Tensor, layer: int,
                   obs_ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = z.shape
        eye = torch.eye(N, device=z.device, dtype=torch.bool).unsqueeze(0)
        if self.mode == "static":
            logits = self.static_logits.unsqueeze(0).expand(B, -1, -1)
        else:
            q = self.phi[layer](z)                              # [B,N,d]
            logits = torch.bmm(q, q.transpose(1, 2)) / math.sqrt(q.size(-1))
        if self.availability_mode in ("edge_bias", "both"):
            if obs_ratio is None:
                raise ValueError(
                    "availability_mode requires obs_ratio; encoder must supply it")
            # ri = 目的端 i（行），rj = 源端 j（列）（S2-2 审查修正广播方向）
            ri = obs_ratio.unsqueeze(-1).expand(B, N, N)        # [B,N,N] 行不变
            rj = obs_ratio.unsqueeze(-2).expand(B, N, N)        # [B,N,N] 列不变
            feats = torch.stack([ri, rj, ri * rj], dim=-1)      # [B,N,N,3]
            bias = self.edge_bias[layer](feats).squeeze(-1)
            # 可用性偏置在 softmax 前加到 logit 空间（对角不加，self-loop 稳定）
            logits = logits + bias.masked_fill(eye, 0.0)
        if self.self_loop:
            logits = logits.masked_fill(eye, float("-inf"))
            adj = torch.softmax(logits, dim=-1)
            adj = adj.masked_fill(eye, 0.0)
            adj = adj + eye.to(adj.dtype)
        else:
            adj = torch.softmax(logits, dim=-1)
        if self.availability_mode in ("source_suppress", "both"):
            if obs_ratio is None:
                raise ValueError(
                    "availability_mode requires obs_ratio; encoder must supply it")
            # 源端抑制后对整行（含 self-loop）重归一化，保持行分布契约；
            # self_loop=True 时行和恒 >= 1（对角存活），无零行退化。
            adj = self._suppress_unavailable_sources(adj, obs_ratio)
        _check_finite(adj, f"graph.adjacency[layer{layer}]")
        return adj

    def forward(self, z: torch.Tensor,
                obs_ratio: Optional[torch.Tensor] = None
                ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """返回 (A_首层邻接或 None, 融合前的 cross-sensor 表示)。"""
        if self.mode == "none" or self.n_layers == 0:
            return None, z
        h = z
        first_adj = None
        for layer in range(self.n_layers):
            adj = self._adjacency(h, layer, obs_ratio)
            if first_adj is None:
                first_adj = adj
            msg = torch.bmm(adj, self.w[layer](h))
            h = self.norm[layer](h + msg)                       # 残差
        _check_finite(h, "graph.propagated")
        return first_adj, h


class GatedFusion(nn.Module):
    """self-history 与 cross-sensor 表示的门控融合。

    z_out = alpha * z_self + (1 - alpha) * z_cross，
    alpha = sigmoid(Linear([z_self; z_cross; cond_emb]))。

    模式：
    - gated     ：完整门控（默认）
    - add       ：直接相加（A7 消融）
    - self_only ：只用 self-history（A5 消融）
    - no_graph  ：等价于不使用跨传感器分支（与 self_only 输出一致，图模块被跳过）
    """

    MODES = ("gated", "add", "self_only", "no_graph")

    def __init__(self, d_model: int, condition_dim: int = 0, mode: str = "gated",
                 use_availability: bool = False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown fusion mode: {mode}")
        self.mode = mode
        self.use_availability = use_availability
        self.needs_graph = mode in ("gated", "add", "self_only")
        if mode == "gated":
            # use_availability 时把目的端通道可用性 r_i 拼进门输入（+1 维）
            extra = 1 if use_availability else 0
            self.gate = nn.Linear(2 * d_model + condition_dim + extra, d_model)

    def forward(
        self,
        z_self: torch.Tensor,                    # [B,C,d]
        z_cross: Optional[torch.Tensor],         # [B,C,d]
        cond_emb: Optional[torch.Tensor] = None, # [B,d_c]
        obs_ratio: Optional[torch.Tensor] = None,# [B,C]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.needs_graph or z_cross is None:
            return z_self, None
        if self.mode == "self_only":
            return z_self, None
        if self.mode == "add":
            return z_self + z_cross, None
        parts = [z_self, z_cross]
        if cond_emb is not None:
            if cond_emb.dim() == 2:
                cond_emb = cond_emb.unsqueeze(1).expand(-1, z_self.size(1), -1)
            parts.append(cond_emb)
        if self.use_availability:
            if obs_ratio is None:
                raise ValueError(
                    "fusion use_availability requires obs_ratio from encoder")
            parts.append(obs_ratio.unsqueeze(-1))
        gate_in = torch.cat(parts, dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))               # [B,C,d]
        return alpha * z_self + (1.0 - alpha) * z_cross, alpha


class MTKDGEncoder(nn.Module):
    """多尺度时间核 + 动态传感器图编码器（第三章统一表示）。"""

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_kernels: int = 4,
        scales: Sequence[float] = DEFAULT_SCALES,
        graph_mode: str = "dynamic",
        graph_layers: int = 1,
        graph_self_loop: bool = True,
        fusion_mode: str = "gated",
        share_kernels_across_scales: bool = False,
        condition_dim: int = 0,
        dropout: float = 0.1,
        availability_mode: str = "none",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_kernels = n_kernels
        self.scales = tuple(sorted(set(float(s) for s in scales)))
        if not (0.0 < self.scales[0] <= 1.0):
            raise ValueError("scales must be within (0, 1]")
        self.model_version = MODEL_VERSION

        # 每尺度独立时间核（或共享，配置决定）
        n_scale_modules = 1 if share_kernels_across_scales else len(self.scales)
        self.share_kernels = share_kernels_across_scales
        self.kernels = nn.ModuleList(
            [TimeKernelPooling(n_kernels) for _ in range(n_scale_modules)]
        )
        self.scale_proj = nn.ModuleList(
            [nn.Linear(n_kernels + 1, d_model) for _ in range(len(self.scales))]
        )
        self.scale_gate = ScaleGate(n_channels, len(self.scales))
        self.scale_norm = nn.LayerNorm(d_model)

        self.channel_embedding = nn.Embedding(n_channels, d_model)

        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, d_model)

        self.graph = DynamicSensorGraph(
            n_nodes=n_channels,
            d_model=d_model,
            mode=graph_mode,
            n_layers=graph_layers,
            self_loop=graph_self_loop,
            availability_mode=availability_mode,
        )
        # 融合门接收的是已投影的 cond_emb（d_model 维），不是原始 condition 输入
        self.fusion = GatedFusion(
            d_model,
            condition_dim=(d_model if condition_dim > 0 else 0),
            mode=fusion_mode,
            use_availability=(availability_mode != "none"),
        )
        self.fusion_mode = fusion_mode
        self.availability_mode = availability_mode
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _scale_slice(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, fraction: float):
        L = x.size(1)
        n_keep = max(1, int(math.ceil(L * fraction)))
        return x[:, -n_keep:, :], t[:, -n_keep:], mask[:, -n_keep:, :]

    def forward(
        self,
        x: torch.Tensor,                       # [B,H,C]
        t: torch.Tensor,                       # [B,H]
        mask: Optional[torch.Tensor] = None,   # [B,H,C]
        condition: Optional[torch.Tensor] = None,  # [B,H,3]（FD002/FD004 工况）
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.size(-1) != self.n_channels:
            raise ValueError(f"expected x [B,H,{self.n_channels}], got {tuple(x.shape)}")
        if mask is None:
            mask = torch.ones_like(x)
        _check_finite(x, "encoder.input")

        cond_emb = None
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError("encoder configured with condition_dim>0 but condition is None")
            cond_mean = condition.mean(dim=1)                   # [B, cond_dim]
            cond_emb = self.condition_proj(cond_mean)           # [B, d]

        scale_feats = []
        scale_summaries = []
        obs_ratios = []
        for i, fraction in enumerate(self.scales):
            xs, ts, ms = self._scale_slice(x, t, mask, fraction)
            kernel = self.kernels[0] if self.share_kernels else self.kernels[i]
            h, obs_ratio = kernel(xs, ts, ms)                   # [B,C,K], [B,C]
            feat = torch.cat([h, obs_ratio.unsqueeze(-1)], dim=-1)  # [B,C,K+1]
            scale_feats.append(self.scale_proj[i](feat))        # [B,C,d]
            scale_summaries.append(
                torch.cat([h.mean(dim=1), obs_ratio], dim=-1)   # [B, K+1]
            )
            obs_ratios.append(obs_ratio)

        # MC-Graph：跨尺度平均的逐通道可用性（时间核池化的现成输出），
        # 条件化动态图边权/源端抑制与融合门
        agg_obs_ratio = torch.stack(obs_ratios, dim=0).mean(dim=0)  # [B,C]

        stacked = torch.stack(scale_feats, dim=1)               # [B,S,C,d]
        x_stats = torch.cat(
            [x.mean(dim=1), mask.mean(dim=1)], dim=-1           # [B, 2C]
        )
        gate_weights = self.scale_gate(torch.stack(scale_summaries, dim=1), x_stats)  # [B,S]
        z_self = (stacked * gate_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)     # [B,C,d]
        z_self = self.dropout(self.scale_norm(z_self))

        ids = torch.arange(self.n_channels, device=x.device)
        z_self = z_self + self.channel_embedding(ids).unsqueeze(0)

        adjacency, z_cross = self.graph(z_self, obs_ratio=agg_obs_ratio)
        z_out, alpha = self.fusion(z_self, z_cross, cond_emb, obs_ratio=agg_obs_ratio)
        z_out = self.out_norm(self.dropout(z_out))
        _check_finite(z_out, "encoder.output")

        return {
            "z": z_out,                        # [B, C, d] 任务头输入
            "z_self": z_self,
            "z_cross": z_cross,
            "adjacency": adjacency,            # [B,C,C] 或 None
            "gate_weights": gate_weights,      # [B,S]
            "fusion_alpha": alpha,             # [B,C,d] 或 None
            "cond_emb": cond_emb,              # [B,d] 或 None
            "obs_ratio": agg_obs_ratio,        # [B,C] 逐通道可用性
        }
