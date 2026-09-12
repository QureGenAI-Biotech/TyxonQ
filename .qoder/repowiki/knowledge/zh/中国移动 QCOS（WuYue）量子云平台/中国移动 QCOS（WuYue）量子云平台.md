---
kind: external_dependency
name: 中国移动 QCOS（WuYue）量子云平台
slug: qcos-wuyue
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---

通过 China Mobile ecloud 提供的 WuYue 系列设备（如 `WuYue-QPUSim-FullAmpSim`）接入 QCOS 量子计算操作系统。需安装独立 SDK wheel（`wuyue-1.0-py3-none-any.whl`，从 ecloud 控制台「编程框架 → SDK 本地部署」下载），并通过环境变量 `QCOS_ACCESS_KEY` / `QCOS_SECRET_KEY` 或 `access_key`/`secret_key` 参数传入。
- 依赖：可选依赖 `md` 组之外单独安装，Python 版本要求 3.11。
- 认证：基于 access_key/secret_key 的双密钥模式，与 TyxonQ 平台的 API Key 不同。