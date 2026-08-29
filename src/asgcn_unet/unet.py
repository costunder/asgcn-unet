from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.body(x))


class ConvGRUCell(nn.Module):
    """Causal analog state; only the graph front-end is ANN-to-SNN converted."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        merged = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(merged, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(merged, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        expected = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        if state is None or tuple(state.shape) != expected:
            state = torch.zeros(expected, device=x.device, dtype=x.dtype)
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=1)))
        return (1.0 - update) * state + update * candidate


class RecurrentUNetDecoder(nn.Module):
    """Two-level residual U-Net with an optional bottleneck ConvGRU state."""

    def __init__(
        self, input_channels: int, base_channels: int, output_channels: int, recurrent: bool = True
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.enc1 = ResidualBlock(base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4), ResidualBlock(base_channels * 4)
        )
        self.recurrent = ConvGRUCell(base_channels * 4, base_channels * 4) if recurrent else None
        self.up2 = nn.Conv2d(base_channels * 6, base_channels * 2, 3, padding=1)
        self.dec2 = ResidualBlock(base_channels * 2)
        self.up1 = nn.Conv2d(base_channels * 3, base_channels, 3, padding=1)
        self.dec1 = ResidualBlock(base_channels)
        self.head = nn.Conv2d(base_channels, output_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))
        if self.recurrent is not None:
            state = self.recurrent(bottleneck, state)
            bottleneck = bottleneck + state
        u2 = F.interpolate(bottleneck, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        output = torch.sigmoid(self.head(u1))
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return output, state
