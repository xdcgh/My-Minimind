import torch
import torch.nn as nn

from model.config import DCMindConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        y = x.float()
        variance = y.pow(2).mean(-1, keepdim=True)
        y = y * torch.rsqrt(variance + self.eps)

        return (self.weight * y).to(input_dtype)


__all__ = ["DCMindConfig", "RMSNorm"]
