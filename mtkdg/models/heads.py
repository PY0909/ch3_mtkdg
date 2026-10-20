"""任务头与完整点预测模型。

- MetroPT StateHead：输出 [B,P,C]（标准化空间），评估时反归一化到原始单位；
- C-MAPSS RULHead：输出 [B,1] raw-cycle RUL，softplus 保证非负；
- 第三章只有 point loss，无概率头和风险头。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MTKDGConfig
from .mtkdg import MODEL_VERSION, MTKDGEncoder
from .prob_heads import (
    GaussianRULHead,
    GaussianStateHead,
    RULQuantileHead,
    StateQuantileHead,
)

PROB_HEADS = ("quantile", "gaussian", "censored")

STATE_HEAD_VERSION = "metropt_state_head_v1"
RUL_HEAD_VERSION = "rul_head_v1"


class StateHead(nn.Module):
    """MetroPT 未来多步状态头：horizon embedding + 传感器表示 + 全局上下文。

    输入 z [B,C,d]，输出标准化状态预测 [B,P,C]。
    mode="delta"（V4 消融）：输出相对最后观测值的增量，由模型加回 x_last
    （persistence-skip：y = x_last + f(z)，慢趋势通道更易学）。
    """

    def __init__(self, d_model: int, n_channels: int, pred_len: int, mode: str = "direct"):
        super().__init__()
        if mode not in ("direct", "delta"):
            raise ValueError(f"unknown state head mode: {mode}")
        self.mode = mode
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.horizon_embedding = nn.Embedding(pred_len, d_model)
        self.global_proj = nn.Linear(d_model, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, d = z.shape
        P = self.pred_len
        g = self.global_proj(z.mean(dim=1)).view(B, 1, 1, d).expand(B, C, P, d)
        h = self.horizon_embedding.weight.view(1, 1, P, d).expand(B, C, P, d)
        zs = z.unsqueeze(2).expand(B, C, P, d)
        out = self.decoder(torch.cat([zs, h, g], dim=-1)).squeeze(-1)   # [B,C,P]
        return out.transpose(1, 2).contiguous()                          # [B,P,C]（delta 模式下为增量）


class RULHead(nn.Module):
    """C-MAPSS scalar RUL 头：传感器池化 + 工况条件 + 非负输出。

    输出 [B,1] raw-cycle RUL（softplus 非负）。loss 在标准化空间计算
    ((rul_raw - mean) / std 对比标准化标签），由 runner 完成。
    """

    def __init__(self, d_model: int, rul_mean: float, rul_std: float):
        super().__init__()
        self.rul_mean = float(rul_mean)
        self.rul_std = float(rul_std)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # bias 初始化到 softplus^-1(mean)，让初始输出接近 train 平均 RUL
        inv = math.log(math.expm1(max(self.rul_mean, 1e-3)))
        nn.init.constant_(self.mlp[-1].bias, inv)
        nn.init.constant_(self.mlp[-1].weight, 0.0)

    def forward(self, z: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)  # [B,2d]
        d_model = pooled.size(-1) // 2
        if cond_emb is not None:
            feats = torch.cat([pooled, cond_emb], dim=-1)
        else:
            # 工况分支关闭时（A6 消融）用零向量占位，保持维度一致
            zero_pad = torch.zeros(z.size(0), d_model, device=z.device, dtype=z.dtype)
            feats = torch.cat([pooled, zero_pad], dim=-1)
        if not torch.isfinite(feats).all():
            raise RuntimeError("non-finite values in RUL head input")
        return F.softplus(self.mlp(feats))                                 # [B,1] 非负


class MTKDGPointModel(nn.Module):
    """MTK-DG encoder + 数据集专属点预测头。

    RUL 头模式（Loop-2 预注册消融）：
    - single     ：capped 头（主协议 V0）；
    - raw_single ：单头直接监督 raw RUL（P2a 对照：分离"raw 监督"与"多任务"贡献）；
    - dual       ：capped 头 + raw 辅助头，loss=MSE_capped+w*MSE_raw，
                   test 用 raw 头输出（P2b：对准 G4 capped 尾部问题）。
    """

    def __init__(
        self,
        task: str,
        n_channels: int,
        pred_len: int,
        model_cfg: MTKDGConfig,
        rul_mean: float = 0.0,
        rul_std: float = 1.0,
        rul_raw_mean: float = 0.0,
        rul_raw_std: float = 1.0,
    ):
        super().__init__()
        if task not in ("state_forecast", "rul_regression"):
            raise ValueError(f"unknown task: {task}")
        self.task = task
        condition_dim = 3 if task == "rul_regression" else 0
        self.model_version = MODEL_VERSION
        self.encoder = MTKDGEncoder(
            n_channels=n_channels,
            d_model=model_cfg.d_model,
            n_kernels=model_cfg.n_kernels,
            scales=model_cfg.scales,
            graph_mode=model_cfg.graph_mode,
            graph_layers=model_cfg.graph_layers,
            graph_self_loop=model_cfg.graph_self_loop,
            fusion_mode=model_cfg.fusion_mode,
            share_kernels_across_scales=model_cfg.share_kernels_across_scales,
            condition_dim=condition_dim,
            dropout=model_cfg.dropout,
            availability_mode=model_cfg.availability_mode,
        )
        if task == "state_forecast":
            self.state_head_mode = model_cfg.state_head_mode
            self.head = StateHead(model_cfg.d_model, n_channels, pred_len,
                                  mode=model_cfg.state_head_mode)
            self.head_version = STATE_HEAD_VERSION
        else:
            self.rul_head_mode = model_cfg.rul_head
            if model_cfg.rul_head in ("single", "dual"):
                self.head = RULHead(model_cfg.d_model, rul_mean, rul_std)
            if model_cfg.rul_head in ("raw_single", "dual"):
                self.aux_head = RULHead(model_cfg.d_model, rul_raw_mean, rul_raw_std)
            self.head_version = RUL_HEAD_VERSION

    def _enc(self, x, t, mask, condition):
        return self.encoder(x, t, mask, condition)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """test-time 输出：single→capped；raw_single/dual→raw（预注册）。"""
        enc = self._enc(x, t, mask, condition)
        if self.task == "state_forecast":
            out = self.head(enc["z"])                    # [B,P,C]（delta 模式为增量）
            if self.state_head_mode == "delta":
                out = out + x[:, -1, :].unsqueeze(1)     # persistence-skip
            return out
        if self.rul_head_mode == "single":
            return self.head(enc["z"], enc["cond_emb"])
        return self.aux_head(enc["z"], enc["cond_emb"])

    def forward_train(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> dict:
        """训练期输出各监督头的预测（runner 按模式组合 loss）。"""
        enc = self._enc(x, t, mask, condition)
        if self.task == "state_forecast":
            out = self.head(enc["z"])
            if self.state_head_mode == "delta":
                out = out + x[:, -1, :].unsqueeze(1)
            return {"state": out}
        out = {}
        if self.rul_head_mode in ("single", "dual"):
            out["capped"] = self.head(enc["z"], enc["cond_emb"])
        if self.rul_head_mode in ("raw_single", "dual"):
            out["raw"] = self.aux_head(enc["z"], enc["cond_emb"])
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
