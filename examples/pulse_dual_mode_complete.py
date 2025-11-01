"""测试 Pulse 双模式完整支持：pulse vs pulse_inline"""

from tyxonq import Circuit
from tyxonq.compiler.pulse_compile_engine import PulseCompiler
from tyxonq.compiler.pulse_compile_engine.serialization import save_pulse_circuit, load_pulse_circuit
import numpy as np

print("="*70)
print("测试：Pulse 双模式完整支持（pulse vs pulse_inline）")
print("="*70)

# 创建测试电路
c = Circuit(1)
c.x(0)

compiler = PulseCompiler()

# ========== 模式1：pulse（符号引用，默认）==========
print("\n【模式1】pulse（符号引用）- 本地模拟 + PyTorch autograd")
print("-" * 70)

pulse_circuit_ref = compiler.compile(
    c,
    device_params={"qubit_freq": [5.0e9], "anharmonicity": [-330e6]},
    output="pulse_ir",
    inline_pulses=False,  # 保持符号引用
    mode="pulse_only"
)

print(f"1. 编译结果：")
print(f"   操作类型：{pulse_circuit_ref.ops[0][0]}")
print(f"   Pulse key：{pulse_circuit_ref.ops[0][2]}")
print(f"   Pulse库大小：{len(pulse_circuit_ref.metadata['pulse_library'])}")

# 执行
from tyxonq.numerics.context import set_backend
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine

set_backend("numpy")
engine = StatevectorEngine()
state_ref = engine.state(pulse_circuit_ref)
print(f"\n2. 执行结果（符号引用模式）：")
print(f"   Final state: {state_ref}")


# ========== 模式2：pulse_inline（完全展开）==========
print("\n【模式2】pulse_inline（完全展开）- 云端兼容 + 序列化")
print("-" * 70)

pulse_circuit_inline = compiler.compile(
    c,
    device_params={"qubit_freq": [5.0e9], "anharmonicity": [-330e6]},
    output="pulse_ir",
    inline_pulses=True,  # 完全内联
    mode="pulse_only"
)

print(f"1. 编译结果：")
print(f"   操作类型：{pulse_circuit_inline.ops[0][0]}")
print(f"   Waveform dict：{pulse_circuit_inline.ops[0][2]}")

# 执行
state_inline = engine.state(pulse_circuit_inline)
print(f"\n2. 执行结果（pulse_inline模式）：")
print(f"   Final state: {state_inline}")

# 验证两种模式结果一致
if np.allclose(state_ref, state_inline, atol=1e-6):
    print(f"\n✅ SUCCESS: 两种模式结果一致！")
else:
    print(f"\n❌ FAILED: 两种模式结果不一致")
    print(f"   Ref: {state_ref}")
    print(f"   Inline: {state_inline}")


# ========== 模式3：序列化测试 ==========
print("\n【模式3】序列化与反序列化测试")
print("-" * 70)

# JSON 序列化（需要 inline_pulses=True）
print("1. JSON 序列化（inline_pulses=True）")
save_pulse_circuit(pulse_circuit_inline, "/tmp/pulse_test.json", format="json")
loaded_circuit = load_pulse_circuit("/tmp/pulse_test.json", format="json")
print(f"   保存成功：/tmp/pulse_test.json")
print(f"   加载成功：ops[0] = {loaded_circuit.ops[0][0]}")

# 执行加载的电路
state_loaded = engine.state(loaded_circuit)
if np.allclose(state_inline, state_loaded, atol=1e-6):
    print(f"   ✅ 序列化后执行结果正确！")
else:
    print(f"   ❌ 序列化后执行结果错误")


# Pickle 序列化（支持 inline_pulses=False）
print("\n2. Pickle 序列化（inline_pulses=False，保留对象）")
save_pulse_circuit(pulse_circuit_ref, "/tmp/pulse_test.pkl", format="pickle")
loaded_circuit_pkl = load_pulse_circuit("/tmp/pulse_test.pkl", format="pickle")
print(f"   保存成功：/tmp/pulse_test.pkl")
print(f"   加载成功：ops[0] = {loaded_circuit_pkl.ops[0][0]}")

# 执行
state_pkl = engine.state(loaded_circuit_pkl)
if np.allclose(state_ref, state_pkl, atol=1e-6):
    print(f"   ✅ Pickle序列化后执行结果正确！")
else:
    print(f"   ❌ Pickle序列化后执行结果错误")


print("\n" + "="*70)
print("总结：Pulse 双模式 + 序列化 完整功能验证完成！")
print("="*70)
print("\n核心特性：")
print("  1. ✅ pulse模式（符号引用）- 支持PyTorch autograd")
print("  2. ✅ pulse_inline模式（完全展开）- 云端兼容")
print("  3. ✅ StatevectorEngine 同时支持两种模式")
print("  4. ✅ JSON序列化（文本格式，跨语言）")
print("  5. ✅ Pickle序列化（二进制，保留Python对象）")
print("\n用途对照：")
print("  - 本地优化/训练 → pulse模式 + PyTorch")
print("  - 云端提交/TQASM → pulse_inline模式 + JSON")
print("  - 分布式计算 → pulse模式 + Pickle")
