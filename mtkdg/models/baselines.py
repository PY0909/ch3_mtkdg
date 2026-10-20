"""Baseline encoders（评审 Round1 + 实验矩阵 §8.2 要求：统一协议对比）。

设计合同与 MTK-DG encoder 一致：forward(x[B,H,C], t[B,H], mask[B,H,C], condition) -> z[B,C,D]。
任务头（StateHead / RULHead）与训练/评估协议与 MTK-DG 完全共享，
因此 baseline 与主模型的差异被隔离在时间表示编码器内部。

- TCN / GRU / PatchTST / TimesNet：channel-independent（每通道独立建模时间）；
- iTransformer：variate-token（通道即 token，天然输出 [B,C,D]）。
- FD 的 operating settings 已包含在输入通道前 3 列（与主模型同一特征 schema）；
  baseline 不再有额外工况分支（MTK-DG 的工况分支贡献由消融 A6 量化）。

实现参考公开结构（非逐行复制 /home/cit 下任何仓库代码）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import RULHead, StateHead, RUL_HEAD_VERSION, STATE_HEAD_VERSION

BASELINE_NAMES = ("tcn", "gru", "patchtst", "timesnet", "itransformer")
BASELINE_VERSION = "ch3_baselines_v1"


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite values detected in {name}")


class ChannelIndependentEncoder(nn.Module):
    """channel-independent 基类：把 [B,H,C] 视作 B*C 条独立单变量序列。

    子类实现 `_encode_series(x[B*C, H]) -> h[B*C, D]`。
    """

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, t=None, mask=None, condition=None):
        B, H, C = x.shape
        if C != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {C}")
        h = self._encode_series(x.permute(0, 2, 1).reshape(B * C, H))  # [B*C, D]
        _check_finite(h, f"{self.__class__.__name__}.series")
        z = self.out(h).view(B, C, self.d_model)
        return z


class TCNEncoder(ChannelIndependentEncoder):
    """因果膨胀卷积栈（dilation 1,2,4,...），channel-independent 扁平批处理。

    [B,H,C] 展平为 B*C 条单变量序列过同一组小卷积（一次 kernel launch），
    残差 + LayerNorm，末时刻特征输出。
    """

    def __init__(self, n_channels: int, d_model: int, n_layers: int = 4, kernel_size: int = 3):
        super().__init__(n_channels, d_model)
        blocks = []
        self.in_conv = nn.Conv1d(1, d_model, kernel_size, padding=kernel_size - 1)
        for i in range(n_layers - 1):
            dilation = 2 ** i
            blocks.append(nn.Conv1d(
                d_model, d_model, kernel_size,
                dilation=dilation, padding=(kernel_size - 1) * dilation,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def _encode_series(self, s):  # s [N, H]
        H = s.size(-1)
        h = F.gelu(self.in_conv(s.unsqueeze(1)))[..., :H]             # 因果裁剪
        h = self.norms[0](h.transpose(1, 2)).transpose(1, 2)
        for i, conv in enumerate(self.blocks):
            y = F.gelu(conv(h))[..., :H]
            h = self.norms[i + 1]((h + y).transpose(1, 2)).transpose(1, 2)  # 残差
        return h[..., -1]                                              # 末时刻 [N,D]


class GRUEncoder(ChannelIndependentEncoder):
    """双层 GRU，取末时刻隐状态。"""

    def __init__(self, n_channels: int, d_model: int, n_layers: int = 2):
        super().__init__(n_channels, d_model)
        self.gru = nn.GRU(input_size=1, hidden_size=d_model, num_layers=n_layers,
                          batch_first=True, dropout=0.0 if n_layers == 1 else 0.1)

    def _encode_series(self, s):  # s [N, H]
        out, _ = self.gru(s.unsqueeze(-1))      # [N, H, D]
        return out[:, -1]


class PatchTSTEncoder(ChannelIndependentEncoder):
    """Patch 化 + Transformer encoder（channel-independent）。"""

    def __init__(self, n_channels: int, d_model: int, history_len: int,
                 n_layers: int = 2, n_heads: int = 4, patch_len: int = 12, stride: int = 6):
        super().__init__(n_channels, d_model)
        if history_len < patch_len:
            patch_len = max(4, history_len // 4)
            stride = max(2, patch_len // 2)
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (history_len - patch_len) // stride + 1
        self.n_patches = n_patches
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.flatten = nn.Linear(n_patches * d_model, d_model)

    def _encode_series(self, s):  # s [N, H]
        patches = s.unfold(1, self.patch_len, self.stride)     # [N, P, patch_len]
        if patches.size(1) != self.n_patches:
            # history_len 与构建时不一致时按实际数量裁齐
            patches = patches[:, : self.n_patches]
            if patches.size(1) < self.n_patches:
                pad = patches.new_zeros(patches.size(0), self.n_patches - patches.size(1), self.patch_len)
                patches = torch.cat([patches, pad], dim=1)
        h = self.embed(patches) + self.pos[:, : patches.size(1)]
        h = self.encoder(h)                                     # [N, P, D]
        return self.flatten(h.reshape(h.size(0), -1))


class TimesNetEncoder(ChannelIndependentEncoder):
    """简化 TimesNet：FFT 主周期 → 2D 折叠卷积 → 展开回序列 → 1D 卷积摘要。

    保留核心机制（周期检测 + 周期内/周期间 2D 建模）；
    序列按行扁平（每行一条单变量序列），周期相同的行共享一次批量 2D 卷积，
    避免逐样本 Python 循环。
    """

    def __init__(self, n_channels: int, d_model: int, top_k: int = 1):
        super().__init__(n_channels, d_model)
        self.top_k = top_k
        self.two_d = nn.Sequential(
            nn.Conv2d(1, d_model // 2, (3, 3), padding=1), nn.GELU(),
            nn.Conv2d(d_model // 2, 1, (3, 3), padding=1),
        )
        self.summary = nn.Sequential(
            nn.Conv1d(1, d_model // 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, 3, padding=1), nn.GELU(),
        )

    def _encode_series(self, s):  # s [N, H]
        N, H = s.shape
        fft = torch.fft.rfft(s, dim=1)
        amp = fft.abs()
        amp[:, 0] = 0.0                                            # 去直流
        k = min(self.top_k, max(1, amp.size(1) - 1))
        period = torch.topk(amp[:, 1:], k, dim=1).indices[:, 0] + 1    # [N]
        fused = torch.zeros_like(s)
        for p in torch.unique(period).tolist():                    # 唯一周期数量有限
            p = max(2, min(int(p), H))
            sel = (period == p)
            if not sel.any():
                continue
            sub = s[sel]                                           # [n, H]
            reps = math.ceil(H / p)
            padded = sub.new_zeros(sub.size(0), reps * p)
            padded[:, :H] = sub
            grid = padded.view(-1, 1, reps, p)                     # [n,1,R,P]
            g = self.two_d(grid).reshape(sub.size(0), -1)[:, :H]   # 展开回时间轴
            fused[sel] = g
        return self.summary((s + fused).unsqueeze(1))[..., -1]     # [N,D]


class ITransformerEncoder(nn.Module):
    """iTransformer：通道即 token，注意力在变量轴上；天然输出 [B,C,D]。"""

    def __init__(self, n_channels: int, d_model: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.embed = nn.Linear(1, d_model)   # 每个通道的整条序列按位置嵌入
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, t=None, mask=None, condition=None):
        B, H, C = x.shape
        if C != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {C}")
        tokens = self.embed(x.permute(0, 2, 1).unsqueeze(-1))   # [B,C,H,1]->[B,C,H,D] -> embed over last dim
        tokens = tokens.mean(dim=2)                             # 通道 token [B,C,D]
        h = self.encoder(tokens)
        _check_finite(h, "itransformer.out")
        return self.out(h)


class BaselinePointModel(nn.Module):
    """baseline encoder + 与 MTK-DG 完全相同的任务头。"""

    def __init__(
        self,
        task: str,
        encoder_name: str,
        n_channels: int,
        pred_len: int,
        d_model: int = 64,
        history_len: int = 50,
        rul_mean: float = 0.0,
        rul_std: float = 1.0,
    ):
        super().__init__()
        if task not in ("state_forecast", "rul_regression"):
            raise ValueError(f"unknown task: {task}")
        if encoder_name not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline: {encoder_name}, choose from {BASELINE_NAMES}")
        self.task = task
        self.encoder_name = encoder_name
        self.model_version = f"{BASELINE_VERSION}:{encoder_name}"
        if encoder_name == "tcn":
            self.encoder = TCNEncoder(n_channels, d_model)
        elif encoder_name == "gru":
            self.encoder = GRUEncoder(n_channels, d_model)
        elif encoder_name == "patchtst":
            self.encoder = PatchTSTEncoder(n_channels, d_model, history_len)
        elif encoder_name == "timesnet":
            self.encoder = TimesNetEncoder(n_channels, d_model)
        else:
            self.encoder = ITransformerEncoder(n_channels, d_model)
        if task == "state_forecast":
            self.head = StateHead(d_model, n_channels, pred_len)
            self.head_version = STATE_HEAD_VERSION
        else:
            self.head = RULHead(d_model, rul_mean, rul_std)
            self.head_version = RUL_HEAD_VERSION

    def forward(self, x, t=None, mask=None, condition=None) -> torch.Tensor:
        z = self.encoder(x, t, mask, condition)                 # [B,C,D]
        if self.task == "state_forecast":
            return self.head(z)
        return self.head(z, None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
