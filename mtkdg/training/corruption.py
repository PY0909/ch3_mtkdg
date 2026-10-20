"""MC-Graph 鲁棒训练的在线 corruption 生成器（GPU 张量，无 CPU 往返）。

设计约束：
- 只作用于传感器通道；FD 的工况通道（前 condition_dim 列）永不损坏；
- 通道级与时间块级两种 corruption 可叠加（mixed）；
- 每个 batch 以 corrupt_prob 概率决定是否施加（预注册超参，不读 val/test）；
- corruption 同时改写 x（置零）与 mask（供时间核池化的 obs_ratio 感知）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _rand(shape, device, generator=None):
    """generator（CPU）与目标 device 解耦：CPU 采样后搬运，避免设备不匹配。"""
    if generator is not None:
        return torch.rand(shape, generator=generator).to(device)
    return torch.rand(shape, device=device)


def channel_dropout(
    x: torch.Tensor, mask: torch.Tensor, rate: float,
    protect_channels: int = 0, generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """整条历史随机屏蔽部分传感器通道。"""
    if rate <= 0:
        return x, mask
    B, H, C = x.shape
    keep = (_rand((B, C), x.device, generator) >= rate)
    if protect_channels > 0:
        keep[:, :protect_channels] = True
    keep = keep.unsqueeze(1).to(x.dtype)               # [B,1,C]
    return x * keep, mask * keep


def time_block_mask(
    x: torch.Tensor, mask: torch.Tensor, ratio: float,
    protect_channels: int = 0, max_blocks: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """连续时间块置零（全通道），块数 <= max_blocks。"""
    if ratio <= 0:
        return x, mask
    B, H, C = x.shape
    keep = torch.ones(B, H, 1, device=x.device, dtype=x.dtype)
    blen = max(1, int(round(H * ratio)))
    for _ in range(max_blocks):
        if blen >= H:
            keep = torch.zeros_like(keep)
            break
        if generator is not None:
            starts = torch.randint(0, H - blen + 1, (B,), generator=generator).to(x.device)
        else:
            starts = torch.randint(0, H - blen + 1, (B,), device=x.device)
        rows = torch.arange(H, device=x.device).unsqueeze(0)          # [1,H]
        in_block = (rows >= starts.unsqueeze(1)) & (rows < (starts.unsqueeze(1) + blen))
        keep = keep * (~in_block).unsqueeze(-1).to(x.dtype)
    if protect_channels > 0 and C > protect_channels:
        # 工况通道不受时间块影响（block 只作用于传感器通道）
        full = torch.ones(B, H, C, device=x.device, dtype=x.dtype)
        full[:, :, protect_channels:] = keep[:, :, 0].unsqueeze(-1)
        keep = full
    return x * keep, mask * keep


def corrupt_batch(
    x: torch.Tensor, mask: torch.Tensor, cfg,
    protect_channels: int, generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按配置生成混合 corruption；cfg 为 MTKDGConfig。

    注意：generator（若提供）必须是 CPU generator（内部采样后搬运到 x.device）；
    缺省 None 时使用全局 RNG（set_seed 已固化，训练路径使用此方式）。
    """
    if not (0 <= protect_channels <= x.size(-1)):
        raise ValueError(
            f"protect_channels must be within [0, {x.size(-1)}], got {protect_channels}")
    draw = _rand((), x.device, generator)
    if draw.item() > cfg.corrupt_prob:
        return x, mask
    if cfg.corrupt_channel_rate > 0:
        x, mask = channel_dropout(x, mask, cfg.corrupt_channel_rate,
                                  protect_channels, generator)
    if cfg.corrupt_block_ratio > 0:
        x, mask = time_block_mask(x, mask, cfg.corrupt_block_ratio,
                                  protect_channels, generator=generator)
    return x, mask
