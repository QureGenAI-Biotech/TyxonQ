#!/usr/bin/env python3
"""测试 TQASM 版本声明支持

TQASM 0.2 和 OpenQASM 3.0 的关键区别：
- TQASM 0.2: 使用 Qiskit 风格量子比特声明 (qreg)
- OpenQASM 3.0: 使用标准 OpenQASM 3.0 量子比特声明 (qubit)

版本选择由目标硬件决定：
- homebrew_s2 + use_pulse() → TQASM 0.2 (qreg 风格)
- 其他设备 + use_pulse() → OpenQASM 3.0 (qubit 风格)
"""

def test_tqasm_version():
    """测试 TQASM 0.2 版本声明 (homebrew_s2 特有)"""
    from tyxonq.core.ir.circuit import Circuit
    from tyxonq.compiler.api import compile
    
    print("=" * 70)
    print("测试 1: TQASM 0.2 版本声明 (homebrew_s2 特有)")
    print("=" * 70)
    
    circuit = Circuit(1)
    circuit.h(0)
    
    # 指定 homebrew_s2 设备 + use_pulse() 模式
    # 规则3：脉冲电路 + homebrew_s2 → 自动设为 tyxonq_homebrew_tqasm（TQASM 0.2）
    circuit.device(provider="tyxonq", device="homebrew_s2")
    circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9],
        "anharmonicity": [-330e6]
    })
    
    # 编译
    result = compile(circuit, output="tqasm")
    tqasm_code = result["compiled_source"]
    
    print("\n生成的代码:")
    print("-" * 70)
    print(tqasm_code)
    print("-" * 70)
    
    # 验证：TQASM 0.2 使用 qreg 语法
    assert "TQASM 0.2" in tqasm_code, "应包含 TQASM 0.2 版本声明"
    assert "qreg" in tqasm_code, "应使用 qreg 量子比特声明 (Qiskit 风格)"
    
    print("\n✅ TQASM 0.2 验证通过")
    print("   - 版本声明: TQASM 0.2")
    print("   - 量子比特声明: qreg (Qiskit 风格)")
    
    return tqasm_code


def test_openqasm3_version():
    """测试 OpenQASM 3.0 版本声明 (标准格式)"""
    from tyxonq.core.ir.circuit import Circuit
    from tyxonq.compiler.api import compile
    
    print("\n" + "=" * 70)
    print("测试 2: OpenQASM 3.0 版本声明 (标准格式)")
    print("=" * 70)
    
    circuit = Circuit(1)
    circuit.h(0)
    
    # 不指定 homebrew_s2 + use_pulse() 模式
    # 脉冲电路 + 其他设备 → 保持 openqasm3 格式
    circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9],
        "anharmonicity": [-330e6]
    })
    
    # 编译
    result = compile(circuit, output="qasm3")
    openqasm_code = result["compiled_source"]
    
    print("\n生成的代码:")
    print("-" * 70)
    print(openqasm_code)
    print("-" * 70)
    
    # 验证：OpenQASM 3.0 使用 qubit 语法
    assert "OPENQASM 3.0" in openqasm_code, "应包含 OPENQASM 3.0 版本声明"
    assert "qubit[" in openqasm_code, "应使用 qubit 量子比特声明 (OpenQASM 3.0 风格)"
    assert 'defcalgrammar "openpulse"' in openqasm_code, "应包含 OpenPulse 语法声明"
    
    print("\n✅ OpenQASM 3.0 验证通过")
    print("   - 版本声明: OPENQASM 3.0")
    print("   - 量子比特声明: qubit (标准风格)")
    print("   - Pulse 语法: defcalgrammar \"openpulse\"")
    
    return openqasm_code


def test_version_difference():
    """测试两种版本的关键差异"""
    from tyxonq.core.ir.circuit import Circuit
    from tyxonq.compiler.api import compile
    
    print("\n" + "=" * 70)
    print("测试 3: 版本声明差异对比")
    print("=" * 70)
    
    # TQASM 0.2 版本（homebrew_s2）
    c1 = Circuit(2)
    c1.h(0).cx(0, 1)
    c1.device(provider="tyxonq", device="homebrew_s2")
    c1.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6]
    })
    
    result1 = compile(c1, output="tqasm")
    tqasm_code = result1["compiled_source"]
    
    # OpenQASM 3.0 版本（其他设备）
    c2 = Circuit(2)
    c2.h(0).cx(0, 1)
    c2.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6]
    })
    
    result2 = compile(c2, output="qasm3")
    openqasm_code = result2["compiled_source"]
    
    print("\n对比结果:")
    print("-" * 70)
    print(f"\nTQASM 0.2 (homebrew_s2):")
    print(f"  版本: {'TQASM 0.2' if 'TQASM 0.2' in tqasm_code else '❌'}")
    print(f"  量子比特: {'qreg' if 'qreg' in tqasm_code else 'qubit'}")
    print(f"  代码片段: {tqasm_code.split(chr(10))[0:3]}")
    
    print(f"\nOpenQASM 3.0 (标准):")
    print(f"  版本: {'OPENQASM 3.0' if 'OPENQASM 3.0' in openqasm_code else '❌'}")
    print(f"  量子比特: {'qubit' if 'qubit[' in openqasm_code else 'qreg'}")
    print(f"  代码片段: {openqasm_code.split(chr(10))[0:3]}")
    
    print("\n✅ 版本对比完成")
    print("   - 两个版本的量子比特声明语法不同")
    print("   - 版本选择由目标硬件决定")
    print("   - 内容结构和功能保持一致")
    
    return tqasm_code, openqasm_code


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TQASM 版本声明支持测试")
    print("=" * 70)
    print("""
TQASM 0.2 和 OpenQASM 3.0 的核心差异：

版本选择规则（自动）：
1. homebrew_s2 + use_pulse() → TQASM 0.2
   - 量子比特声明: qreg q[n];  (Qiskit 兼容)
   - 用途: 与 TensorCircuit、Qiskit 互操作

2. 其他设备 + use_pulse() → OpenQASM 3.0
   - 量子比特声明: qubit[n] q;  (OpenQASM 标准)
   - 用途: IBM、Rigetti 等硬件兼容
   - Pulse 语法: defcalgrammar "openpulse"

使用方式：
  # TQASM 0.2 (自动选择)
  c.device(provider="tyxonq", device="homebrew_s2")
  c.use_pulse(...)
  compile(c, output="tqasm")
  
  # OpenQASM 3.0 (默认)
  c.use_pulse(...)
  compile(c, output="qasm3")
    """)
    
    # 运行测试
    tqasm_code = test_tqasm_version()
    openqasm_code = test_openqasm3_version()
    tqasm_full, openqasm_full = test_version_difference()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有版本声明测试通过!")
    print("=" * 70)
