"""
BitLinear layer with ternary {-1, 0, +1} weight quantization-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_ternary(w: torch.Tensor) -> torch.Tensor:
    gamma = w.abs().mean()
    w_scaled = w / (gamma + 1e-8)
    w_q = w_scaled.round().clamp(-1, 1)
    return (w_q - w).detach() + w


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = qmax / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_q = (x * scale).round().clamp(-qmax, qmax) / scale
    return (x_q - x).detach() + x


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = activation_quant(x) if self.training else x
        w_q = ste_ternary(self.weight)
        return F.linear(x_q, w_q, self.bias)
