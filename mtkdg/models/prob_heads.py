"""第四章候选概率头（试点实现）：全部接在冻结的第三章 MTK-DG encoder 上。

设计合同：输入 z [B,C,d]（encoder 输出），输出标准化空间分布参数；
评估时反归一化到原始单位，在 raw unit 上计算覆盖率/CRPS/区间宽度。

- QuantileHead  ：Q 个分位数（pinball + 轻量 crossing 罚），q34/q90 覆盖直接可读；
- GaussianHead  ：mu/logvar（高斯 NLL），CRPS 用闭式正态公式；
- CensoredHead  ：FD 专用——训练标签 capped@125 视为右删失，
  未删失窗口走高斯 NLL，删失窗口走 log-survival（Tobit 似然）；
  测试时对 raw 目标给分位数覆盖（直接对准第三章诊断出的 >125 失败模式）。

参考谱系：pinball≈CRPS 离散化（Tibshirani 讲义）；Tobit 删失似然
（Deep Tobit, Wu 2026；Jean-Pierre PHM 2024）；高斯 NLL 头（标准实践）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 11 个分位水平：边缘 5%/95%，中部 10%..90%，覆盖指标读 q05-95 (90%) 与 q25-75 (50%)
QUANTILE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PROB_HEAD_VERSION = "ch4_prob_heads_v1"


@dataclass
class ProbabilisticOutput:
    family: str
    parameters: dict
    quantiles: Optional[torch.Tensor]
    target_unit: str
    head_version: str
    support: Optional[torch.Tensor] = None


def quantile_index(level: float) -> int:
    return QUANTILE_LEVELS.index(level)


class StateQuantileHead(nn.Module):
    """MetroPT 概率头：z [B,C,d] -> 分位数 [B,P,C,Q]（标准化空间）。"""

    version = "state_quantile_v1"

    def __init__(self, d_model: int, n_channels: int, pred_len: int):
        super().__init__()
        self.Q = len(QUANTILE_LEVELS)
        self.horizon_embedding = nn.Embedding(pred_len, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, self.Q),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, d = z.shape
        P = self.horizon_embedding.num_embeddings
        g = z.mean(dim=1).view(B, 1, 1, d).expand(B, C, P, d)
        h = self.horizon_embedding.weight.view(1, 1, P, d).expand(B, C, P, d)
        zs = z.unsqueeze(2).expand(B, C, P, d)
        q = self.decoder(torch.cat([zs, h, g], dim=-1))          # [B,C,P,Q]
        out = q.permute(0, 2, 1, 3).contiguous()                  # [B,P,C,Q]
        return _monotone_quantiles(out)


class RULQuantileHead(nn.Module):
    """FD 概率头：z [B,C,d] -> 分位数 [B,Q]（标准化空间，单调化）。"""

    version = "rul_quantile_v1"

    def __init__(self, d_model: int, rul_mean: float = 0.0, rul_std: float = 1.0):
        super().__init__()
        self.Q = len(QUANTILE_LEVELS)
        self.rul_mean, self.rul_std = float(rul_mean), float(rul_std)
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, self.Q),
        )

    def forward(self, z: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        feats = pooled if cond_emb is None else torch.cat([pooled, cond_emb], dim=-1)
        if cond_emb is None:
            feats = torch.cat([feats, torch.zeros_like(pooled)], dim=-1)
        q = self.net(feats)                                       # [B,Q]
        return F.softplus(_monotone_quantiles(q))


class GaussianStateHead(nn.Module):
    """MetroPT 高斯头：输出 [B,P,C] mu 与 logvar（标准化空间）。"""

    version = "state_gaussian_v1"

    def __init__(self, d_model: int, n_channels: int, pred_len: int):
        super().__init__()
        self.horizon_embedding = nn.Embedding(pred_len, d_model)
        self.mu_head = nn.Linear(3 * d_model, 1)
        self.lv_head = nn.Linear(3 * d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, d = z.shape
        P = self.horizon_embedding.num_embeddings
        g = z.mean(dim=1).view(B, 1, 1, d).expand(B, C, P, d)
        h = self.horizon_embedding.weight.view(1, 1, P, d).expand(B, C, P, d)
        zs = z.unsqueeze(2).expand(B, C, P, d)
        feats = torch.cat([zs, h, g], dim=-1)
        mu = self.mu_head(feats).squeeze(-1).permute(0, 2, 1)     # [B,P,C]
        logvar = self.lv_head(feats).squeeze(-1).permute(0, 2, 1).clamp(-8.0, 8.0)
        return torch.stack([mu, logvar], dim=-1)                  # [B,P,C,2]


class GaussianRULHead(nn.Module):
    """FD 高斯头：输出 [B,2] = mu, logvar（标准化空间）。"""

    version = "rul_gaussian_v1"

    def __init__(self, d_model: int, rul_mean: float = 0.0, rul_std: float = 1.0):
        super().__init__()
        self.rul_mean, self.rul_std = float(rul_mean), float(rul_std)
        self.mu_head = nn.Linear(3 * d_model, 1)
        self.lv_head = nn.Linear(3 * d_model, 1)

    def forward(self, z: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        feats = pooled if cond_emb is None else torch.cat([pooled, cond_emb], dim=-1)
        if cond_emb is None:
            feats = torch.cat([feats, torch.zeros_like(pooled)], dim=-1)
        mu = self.mu_head(feats)
        logvar = self.lv_head(feats).clamp(-8.0, 8.0)
        return torch.cat([mu, logvar], dim=-1)                    # [B,2]


# ---------------- losses ----------------

def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor,
                 crossing_weight: float = 0.05) -> torch.Tensor:
    """pred_q [...,Q]，target [...]；返回标量。附轻量 crossing 罚。"""
    levels = pred_q.new_tensor(QUANTILE_LEVELS)
    diff = target.unsqueeze(-1) - pred_q
    loss = torch.maximum(levels * diff, (levels - 1) * diff).mean()
    violations = F.relu(pred_q[..., :-1] - pred_q[..., 1:])
    crossing = violations.mean()
    return loss + crossing_weight * crossing


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mu) ** 2 / logvar.exp() + math.log(2 * math.pi)).mean()


def tobit_nll(mu: torch.Tensor, logvar: torch.Tensor, target_capped: torch.Tensor,
              cap: float) -> torch.Tensor:
    """右删失（Tobit）NLL：未删失走高斯 NLL；删失走 log-survival。

    target_capped 为标准化前的 raw capped 标签（cap 判定在 raw 空间做）。
    mu/logvar 同样是 raw 空间参数（调用方先反标准化或直接在 raw 空间建模）。
    """
    sigma = (0.5 * logvar).exp().clamp_min(1e-4)
    uncensored = target_capped < cap
    z_unc = (target_capped - mu) / sigma
    ll_unc = -0.5 * z_unc ** 2 - logvar - math.log(2 * math.pi)
    # 删失窗口：log P(y_latent > cap) = log(1 - Phi((cap - mu)/sigma))
    z_cap = (cap - mu) / sigma
    # log(1 - Phi(z)) 数值稳定形式：log PHi(-z)（用 softplus 近似尾部）
    ll_cen = F.logsigmoid(-z_cap)
    ll = torch.where(uncensored, ll_unc, ll_cen)
    return -ll.mean()


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """闭式正态 CRPS（Gneiting & Raftery 2007）。"""
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (sigma * (z * (2 * torch.sigmoid(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi))).mean()


def crps_from_quantiles(pred_q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """分位数版 CRPS 近似 = 2 * 平均 pinball（QUANTILE_LEVELS 网格）。"""
    levels = pred_q.new_tensor(QUANTILE_LEVELS)
    diff = target.unsqueeze(-1) - pred_q
    pin = torch.maximum(levels * diff, (levels - 1) * diff).mean(dim=-1)
    return 2.0 * pin.mean()


def _monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained logits to a strictly nondecreasing quantile vector."""
    base = raw[..., :1]
    increments = F.softplus(raw[..., 1:])
    return torch.cat([base, base + increments.cumsum(dim=-1)], dim=-1)
