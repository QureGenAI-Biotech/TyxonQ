---
kind: design
name: 拆除 platforms/qiskit，将 qiskit 互操作与 qasm2 导出归位到 gate_compiler
source: session
category: adr
---

# 拆除 platforms/qiskit，将 qiskit 互操作与 qasm2 导出归位到 gate_compiler

_来源：7981385 → e51d61d 提交周期内记录的编码计划——内容为规划时意图，实现可能滞后或有出入。_

**状态：** accepted

## 背景
`platforms/qiskit` 被误标为「硬件平台」，但 qiskit 只是第三方库；同时 `compiler.py` 的 qasm2 分支绕道 qiskit 且因传参错误静默回退 IR，导致 `homebrew_s2` 等调用方拿不到字符串。qcos 作为真硬件反而没有对应 platform 目录，语义自相矛盾。

## 决策驱动
- 模块语义正确性（platform=硬件方言 vs interop=第三方适配）
- 修复 qasm2 静默回退 bug
- 零外部依赖的纯 exporter 范式对齐 qasm3

## 备选方案
- **保留 platforms/qiskit 并修补 compiler.py 传参** — 优点：改动最小；缺点：继续把第三方库伪装成硬件平台；不解决 qasm2 路径的根本问题
- **拆出 gate_compiler/interop + 新建 GateQASM2Exporter** — 优点：语义清晰：interop 放互操作、exporters 放文本序列化；qasm2 输出不再依赖 qiskit；修复 homebrew_s2 静默回退
- **把 to_qiskit/from_qiskit 留在 platforms/** — 优点：迁移量小；缺点：routing/circuit.draw/simulator driver 等多处共享的符号仍挂在错误的 package 下

## 决策
删除 `src/tyxonq/compiler/platforms/qiskit/`，新建 `gate_compiler/exporters/gate_qasm2_exporter.py`（镜像 qasm3 范式的零依赖 exporter）与 `gate_compiler/interop/qiskit_interop.py`（迁入 dialect.py 全部符号），并把 `compile()` 的 qasm2 分支改为直接调用 exporter、qiskit 分支仅做 IR→QuantumCircuit 转换（transpile 留给 routing）。`platform="qiskit"` 在 api.py 中降级为「归一到核心 + output 默认 qiskit」的别名。

## 影响
`output="qiskit"` 不再自动 transpile（行为变更，需调用方显式 `route=True`）；qasm2 输出现在总是字符串而非 IR；新增 `GateQASM2Exporter` 对非 qelib1 门抛 ValueError，要求先分解。