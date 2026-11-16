---
kind: design
name: 引入 agent 包与 Contract 元数据层，以四级反馈阶梯服务 AI/agent
source: session
category: adr
---

# 引入 agent 包与 Contract 元数据层，以四级反馈阶梯服务 AI/agent

_来源：7981385 → e51d61d 提交周期内记录的编码计划——内容为规划时意图，实现可能滞后或有出入。_

**状态：** accepted

## 背景
AI/agent 需要机器可读地查询任意接口的参数域/单位/互斥组/返回形状，并在 <100 µs 内拿到电路是否硬件友好的三态结论与可执行修复建议；同时需要在读电路时主动推送相关物理定理。现有实现无结构化契约、无分层能力声明、无静态分析入口。

## 决策驱动
- 运行时从真实实现反射生成元数据（杜绝手写快照漂移）
- Tier 1 纯函数分析严格 O(G) 且零副作用
- 零破坏红线：不改现有 CompileStep/Circuit.compile 签名/链式语义
- 可选依赖隔离（mcp/pydantic 不进 dependencies）

## 备选方案
- **在 CompileStep 或 build_compile_steps 中注册校验** — 优点：与编译管线集成紧密；缺点：违反红线 1，会破坏 tests_core_module/test_compiler_pipeline.py 的步骤名全等断言
- **给 Circuit.compile/run 加新关键字做校验** — 优点：调用点少；缺点：违反红线 2，破坏 core/ir/circuit.py:806-813 的链式 no-op 短路判定
- **新增独立 tyxonq.agent 包 + Contract 装饰器 + Tier 0-3 阶梯** — 优点：零破坏现有实现；Tier 1 纯函数可被 agent 高频调用；schema/ 产物 gitignored；可选依赖隔离

## 决策
新增叶子包 `src/tyxonq/agent/`，定义 `Contract/Param/Bound/Fact/Support` 四个 frozen dataclass 与 `@contract` 装饰器；通过 `introspect.describe()` 合并 signature/type_hints/docstring/Contract 四路信息；提供 Tier 0（catalog 查询）、Tier 1（metrics/hardware_fit/hints，O(G)、零副作用）、Tier 2（compile 注入指标）、Tier 3（执行）四级接口；Phase 1 只追加 `Circuit.check()/report()/hints()` 三个新方法，不动任何现有方法；所有元数据由 `CATALOG` 清单驱动，`schema/` 产物写入 .gitignore。

## 影响
agent 首次 import 目标模块有延迟，但后续 Tier 1 查询 <100 µs；`source_ref`/`verified_by` 必须可解析，否则门禁测试失败；`DeviceRule` 扩展了 basis_gates/noise_model/t1_us 等可选键；chem 层门面（如 qc_scanner）需逐步补签名与 Contract 注解。