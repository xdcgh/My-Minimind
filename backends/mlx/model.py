import math

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


def precompute_freqs(dim: int, end: int, rope_base: float = 1e6, rope_scaling: dict | None = None):
    assert dim % 2 == 0

    freqs = 1.0 / (rope_base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    attention_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attention_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        if end / orig_max > 1.0:

            def inv_dim(b: float) -> float:
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            ramp = mx.clip(
                (mx.arange(dim // 2).astype(mx.float32) - low) / max(high - low, 0.001),
                0,
                1,
            )

            freqs = freqs * (1 - ramp + ramp / factor)

    t = mx.arange(end)
    freqs = mx.outer(t, freqs).astype(mx.float32)

    freqs_cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1) * attention_factor
    freqs_sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1) * attention_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return mx.concatenate([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], axis=-1)

    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos + rotate_half(q) * sin).astype(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).astype(k.dtype)

    return q_embed, k_embed


__all__ = ["DCMindConfig", "RMSNorm"]
