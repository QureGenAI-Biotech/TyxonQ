---
kind: external_dependency
name: TyxonQ 自有量子硬件云平台（Homebrew_S2）
slug: tyxonq-cloud
category: external_dependency
category_hints:
    - vendor_identity
    - client_constraint
scope:
    - '**'
---

- 认证：API Key 通过 `set_token` 注入；运行期由 platform driver 透传到后端。
- 约束：该 provider 为项目自有服务，非通用第三方 SDK，接入方式与其他云厂商不同。