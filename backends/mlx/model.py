import mlx.core as mx
import mlx.nn as nn

from model.config import DCMindConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        variance = mx.mean(mx.square(y), axis=-1, keepdims=True)
        y = y * mx.rsqrt(variance + self.eps)

        return (self.weight * y).astype(x.dtype)


__all__ = ["DCMindConfig", "RMSNorm"]
