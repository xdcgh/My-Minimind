from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter


def benchmark(name: str, fn: Callable[[], object], warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()

    started_at = perf_counter()
    for _ in range(steps):
        fn()
    elapsed = perf_counter() - started_at

    avg_ms = elapsed / steps * 1000
    print(f"{name}: {avg_ms:.3f} ms/step")
    return avg_ms


def run_torch_rmsnorm(
    hidden_size: int, batch_size: int, seq_len: int, warmup: int, steps: int
) -> None:
    import torch

    from backends.torch.model import RMSNorm

    x = torch.randn(batch_size, seq_len, hidden_size)
    layer = RMSNorm(hidden_size)

    def step() -> torch.Tensor:
        return layer(x)

    benchmark("torch.rmsnorm", step, warmup, steps)


def run_mlx_rmsnorm(
    hidden_size: int, batch_size: int, seq_len: int, warmup: int, steps: int
) -> None:
    import mlx.core as mx

    from backends.mlx.model import RMSNorm

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    layer = RMSNorm(hidden_size)

    def step() -> mx.array:
        y = layer(x)
        mx.eval(y)
        return y

    benchmark("mlx.rmsnorm", step, warmup, steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 PyTorch 与 MLX 后端的基础算子性能")
    parser.add_argument("--backend", choices=["torch", "mlx", "all"], default="all")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend in {"torch", "all"}:
        run_torch_rmsnorm(args.hidden_size, args.batch_size, args.seq_len, args.warmup, args.steps)
    if args.backend in {"mlx", "all"}:
        run_mlx_rmsnorm(args.hidden_size, args.batch_size, args.seq_len, args.warmup, args.steps)


if __name__ == "__main__":
    main()
