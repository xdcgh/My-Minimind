# My-Minimind

跟着视频教程手搓一版 [minimind](https://github.com/jingyaogong/minimind) 的学习仓库。

## 本地参考代码

参考仓库已克隆到本地：

```text
external/minimind
```

该目录仅作为本地参考，已在 `.gitignore` 中忽略，不会提交到本仓库。

## PyTorch / MLX 后端对比

本仓库采用同一套项目结构承载 PyTorch 与 MLX 两个后端，避免复制两份项目导致数据处理、模型配置和 benchmark 逻辑逐渐不一致。

建议分层：

```text
backends/torch/  # PyTorch 相关实现
backends/mlx/    # MLX 相关实现
model/config.py  # 共享模型配置
benchmark/       # 共享性能对比脚本
```

依赖通过 `uv` dependency groups 区分：

```bash
uv sync --group torch
uv sync --group mlx
uv sync --all-groups
```

基础 RMSNorm 对比脚本：

```bash
uv run --group torch --group mlx python -m benchmark.compare_backends
uv run --group torch python -m benchmark.compare_backends --backend torch
uv run --group mlx python -m benchmark.compare_backends --backend mlx
```
