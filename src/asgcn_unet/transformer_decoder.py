"""Hierarchical shifted-window Transformer decoder for controlled ablation.

The two spatial reductions, channel pyramid and six mixing blocks match
``RecurrentUNetDecoder``; its bottleneck ConvGRU and final resize policy are
retained. Spatial mixing is attention plus a channel MLP, not residual CNNs.
Shifted local attention follows the architectural idea in Liu et al., Swin
Transformer (https://arxiv.org/abs/2103.14030), without claiming an exact Swin
reproduction. The 8x8 window is an explicit architecture parameter, not a token
cap: every raster position is processed, including odd-sized image boundaries.
Heads (3,6,12,6,3) give 16 channels/head for the existing 48-channel base;
MLP ratio four is the usual Transformer expansion. These choices preserve the
existing decoder's scale hierarchy, not equal parameter count or measured cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .unet import ConvGRUCell


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B,H,W,C] -> [B,windows,window_size**2,C], no position discarded."""
    b, h, w, c = x.shape
    return (
        x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, (h // window_size) * (w // window_size), window_size**2, c)
    )


def _unpartition(x: torch.Tensor, height: int, width: int, window_size: int) -> torch.Tensor:
    b, _, _, c = x.shape
    return (
        x.reshape(b, height // window_size, width // window_size, window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, height, width, c)
    )


class WindowAttention(nn.Module):
    """All windows/samples/heads are processed in one batched SDPA call.

    Attention has window_size**2 keys, never H*W keys. PyTorch selects its
    available SDPA backend; fused CUDA availability/performance must be measured
    on the actual allocation. A single geometry mask is cached per module, so
    repeated frames do not rebuild static padding/wrap masks. The cache is not
    part of the checkpoint and is replaced, not accumulated, on shape changes.
    """

    def __init__(self, channels: int, heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.channels = _positive_integer(channels, "channels")
        self.heads = _positive_integer(heads, "heads")
        self.window_size = _positive_integer(window_size, "window_size")
        if self.channels % self.heads:
            raise ValueError("channels must be divisible by heads")
        if (
            isinstance(shift_size, bool)
            or not isinstance(shift_size, int)
            or not 0 <= shift_size < self.window_size
        ):
            raise ValueError("shift_size must be an integer in [0, window_size)")
        self.shift_size = shift_size
        self.head_dim = channels // heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(torch.zeros(relative_positions, heads))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)
        differences = coordinates[:, :, None] - coordinates[:, None, :]
        differences = differences + window_size - 1
        relative_index = differences[0] * (2 * window_size - 1) + differences[1]
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        self._mask_key: tuple | None = None
        self._mask_value: tuple[torch.Tensor, torch.Tensor] | None = None

    def _apply(self, fn, recurse: bool = True):
        self._mask_key = None
        self._mask_value = None
        return super()._apply(fn, recurse=recurse)

    def _geometry(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        ws = self.window_size
        padded_h = ((height + ws - 1) // ws) * ws
        padded_w = ((width + ws - 1) // ws) * ws
        shift_h = self.shift_size if padded_h > ws else 0
        shift_w = self.shift_size if padded_w > ws else 0
        key = (height, width, device.type, device.index)
        if key != self._mask_key:
            y, x = torch.meshgrid(
                torch.arange(padded_h, device=device),
                torch.arange(padded_w, device=device),
                indexing="ij",
            )
            valid = (y < height) & (x < width)
            coordinates = torch.stack((y, x), dim=-1).unsqueeze(0)
            valid = valid.unsqueeze(0).unsqueeze(-1)
            if shift_h or shift_w:
                coordinates = torch.roll(coordinates, (-shift_h, -shift_w), (1, 2))
                valid = torch.roll(valid, (-shift_h, -shift_w), (1, 2))
            coordinates = _partition(coordinates, ws)[0]
            valid = _partition(valid, ws)[0, :, :, 0]
            distance = coordinates[:, :, None, :] - coordinates[:, None, :, :]
            # Opposite image borders must not become neighbours after a roll.
            no_wrap = (distance.abs() < ws).all(dim=-1)
            allowed = no_wrap & valid[:, None, :]
            # Discarded padded queries get one finite self entry. Real queries
            # can never read padded keys; this also avoids all-masked softmaxes.
            diagonal = torch.eye(ws * ws, dtype=torch.bool, device=device)
            allowed = torch.where(valid[:, :, None], allowed, diagonal[None])
            self._mask_key = key
            self._mask_value = (allowed, valid)
        assert self._mask_value is not None
        allowed, valid = self._mask_value
        return allowed, valid, padded_h, padded_w, shift_h, shift_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.channels or min(x.shape[:3]) <= 0:
            raise ValueError("WindowAttention expects non-empty [B,H,W,channels]")
        b, h, w, c = x.shape
        ws = self.window_size
        allowed, valid, hp, wp, sh, sw = self._geometry(h, w, x.device)
        padded = F.pad(x, (0, 0, 0, wp - w, 0, hp - h))
        if sh or sw:
            padded = torch.roll(padded, (-sh, -sw), (1, 2))
        windows = _partition(padded, ws)
        count = windows.shape[1]
        length = ws * ws
        qkv = (
            self.qkv(windows)
            .reshape(b * count, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        bias = self.relative_position_bias[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(length, length, self.heads).permute(2, 0, 1).to(qkv.dtype)
        mask = bias.unsqueeze(0).expand(count, -1, -1, -1)
        mask = mask.masked_fill(~allowed[:, None], float("-inf"))
        # Flatten only sample/window batch axes, keeping SDPA's fused-kernel
        # compatible four-dimensional q/k/v contract. Mask cost is linear in
        # raster area for fixed window size; no full-image attention matrix.
        mask = mask.unsqueeze(0).expand(b, -1, -1, -1, -1)
        mask = mask.reshape(b * count, self.heads, length, length)
        mixed = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=mask, dropout_p=0.0
        )
        mixed = mixed.transpose(1, 2).reshape(b, count, length, c)
        mixed = self.projection(mixed) * valid[None, :, :, None].to(mixed.dtype)
        result = _unpartition(mixed, hp, wp, ws)
        if sh or sw:
            result = torch.roll(result, (sh, sw), (1, 2))
        return result[:, :h, :w, :]


class WindowTransformerBlock(nn.Module):
    def __init__(
        self, channels: int, heads: int, window_size: int, shift_size: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        hidden = int(channels * mlp_ratio)
        if hidden < 1:
            raise ValueError("mlp_ratio produces an empty MLP")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowAttention(channels, heads, window_size, shift_size)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Learned 2x2 channel merging; odd borders are padded, never dropped."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels * 4)
        self.reduction = nn.Linear(channels * 4, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[1:3]
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        patches = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1
        )
        return self.reduction(self.norm(patches))


class PatchExpansion(nn.Module):
    """Learned dense 2x2 token expansion, cropped only to undo merge padding."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.expansion = nn.Linear(input_channels, output_channels * 4)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        b, h, w, _ = x.shape
        output_h, output_w = output_size
        if output_h not in (h * 2 - 1, h * 2) or output_w not in (w * 2 - 1, w * 2):
            raise ValueError("PatchExpansion size must undo one padded 2x2 merge")
        x = self.expansion(x).reshape(b, h, w, 2, 2, self.output_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * 2, w * 2, self.output_channels)
        return x[:, :output_h, :output_w]


class RecurrentTransformerDecoder(nn.Module):
    """Two-level window Transformer with the baseline's analog ConvGRU state."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        output_channels: int,
        recurrent: bool = True,
        *,
        depths: Sequence[int] = (1, 1, 2, 1, 1),
        heads: Sequence[int] = (3, 6, 12, 6, 3),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_integer(input_channels, "input_channels")
        self.base_channels = _positive_integer(base_channels, "base_channels")
        self.output_channels = _positive_integer(output_channels, "output_channels")
        self.window_size = _positive_integer(window_size, "window_size")
        if len(depths) != 5 or len(heads) != 5:
            raise ValueError("depths and heads must each specify all five spatial stages")
        self.depths = tuple(_positive_integer(value, "depths") for value in depths)
        self.heads = tuple(_positive_integer(value, "heads") for value in heads)
        if not isinstance(recurrent, bool):
            raise TypeError("recurrent must be boolean")
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)):
            raise TypeError("mlp_ratio must be finite and positive")
        self.mlp_ratio = float(mlp_ratio)
        stage_channels = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
            2 * base_channels,
            base_channels,
        )
        stages = []
        block_index = 0
        for channels, depth, stage_heads in zip(stage_channels, self.depths, self.heads):
            blocks = []
            for _ in range(depth):
                shift = 0 if block_index % 2 == 0 else window_size // 2
                blocks.append(
                    WindowTransformerBlock(
                        channels, stage_heads, window_size, shift, self.mlp_ratio
                    )
                )
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stem = nn.Linear(input_channels, base_channels)
        self.enc1, self.enc2, self.bottleneck, self.dec2, self.dec1 = stages
        self.down1 = PatchMerging(base_channels)
        self.down2 = PatchMerging(base_channels * 2)
        self.recurrent = ConvGRUCell(base_channels * 4, base_channels * 4) if recurrent else None
        self.expand2 = PatchExpansion(base_channels * 4, base_channels * 2)
        self.up2 = nn.Linear(base_channels * 4, base_channels * 2)
        self.expand1 = PatchExpansion(base_channels * 2, base_channels)
        self.up1 = nn.Linear(base_channels * 2, base_channels)
        self.head = nn.Linear(base_channels, output_channels)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 4 or x.shape[1] != self.input_channels or min(x.shape) <= 0:
            raise ValueError("Transformer decoder expects non-empty [B,input_channels,H,W]")
        if len(output_size) != 2:
            raise ValueError("output_size must contain height and width")
        output_size = tuple(_positive_integer(value, "output_size") for value in output_size)
        e1 = self.enc1(self.stem(x.permute(0, 2, 3, 1)))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))
        if self.recurrent is not None:
            raster = bottleneck.permute(0, 3, 1, 2)
            state = self.recurrent(raster, state)
            bottleneck = bottleneck + state.permute(0, 2, 3, 1)
        u2 = self.expand2(bottleneck, e2.shape[1:3])
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=-1)))
        u1 = self.expand1(u2, e1.shape[1:3])
        u1 = self.dec1(self.up1(torch.cat((u1, e1), dim=-1)))
        output = torch.sigmoid(self.head(u1)).permute(0, 3, 1, 2)
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return output, state
