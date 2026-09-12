---
kind: external_dependency
name: Qiskit 路由与 IR 互操作
slug: qiskit
category: external_dependency
category_hints:
    - framework_behavior
    - sdk_real_api
scope:
    - '**'
---

Qiskit 在本项目中**降级为互操作别名**（`translation/` 已并入 `interop/`），不再作为主编译器。仅用于：
- 路由：SABRE 布局 + SWAP 插入委托给 `qiskit.transpile`，在本地索引空间执行，使用可复现 seed 并保持测量顺序；源 IR 不被改写。
- 约束：`opt_level` 与 `optimization_level` 的映射由 GateCompiler 内部协调，不应由上层直接注入 `optimization_level`，否则会破坏 qiskit 的映射逻辑。