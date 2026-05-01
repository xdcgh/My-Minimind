# 仓库协作说明

## 项目定位

- 本项目是跟着视频教程手搓一版 `minimind` 的学习仓库。
- 主要目标是理解并复现核心实现，而不是直接复制参考仓库代码。
- 本地参考代码位于 `external/minimind/`，该目录已被 `.gitignore` 忽略，不参与本仓库提交。

## 开发环境

- 使用 `uv` 管理 Python 项目与依赖。
- Python 版本以 `.python-version` 为准。
- 项目配置主要维护在 `pyproject.toml`。
- 使用 `ruff` 做代码检查和基础格式约束。

## 常用命令

```bash
uv run python main.py
uv run ruff check .
```

## 代码与依赖约定

- 新增 Python 依赖时优先使用 `uv add` 或 `uv add --dev`。
- 修改依赖后需要提交 `pyproject.toml` 和 `uv.lock`。
- 不要把 `external/minimind/` 中的参考仓库内容直接提交进本仓库。
- 如果需要借鉴参考实现，应重新理解后在本项目中手写实现。

## Git 提交规范

- 提交信息使用中文。
- 默认不使用英文 conventional commit 前缀。
- 格式采用“先总后分”：第一行写整体总结，空一行后用分点说明具体改动。

示例：

```text
配置 Ruff 代码检查

- 在 pyproject.toml 中添加 Ruff 基础配置
- 增加 dev 依赖组并锁定 ruff 依赖
- 提交 uv.lock 保持依赖环境可复现
```
