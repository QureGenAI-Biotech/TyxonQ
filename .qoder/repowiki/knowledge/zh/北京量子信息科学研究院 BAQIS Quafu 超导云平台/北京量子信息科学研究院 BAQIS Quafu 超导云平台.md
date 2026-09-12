---
kind: external_dependency
name: 北京量子信息科学研究院 BAQIS Quafu 超导云平台
slug: quafu
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---

接入 BAQIS 提供的 Dongling、Baihua 等超导量子设备。Token 从 https://quafu-sqc.baqis.ac.cn/ 获取，具体接入见 `docs/quafu_provider.md`。
- 依赖：未列入 pyproject 主依赖，需按文档单独配置 token。
- 约束：属于外部高校云平台，token 管理与 TyxonQ/QCOS 均不同。