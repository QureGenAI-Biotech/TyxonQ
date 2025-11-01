#!/usr/bin/env python3
"""
脉冲级量子编程基础示例

本示例展示如何使用 TyxonQ 的脉冲编程功能，直接操控量子比特的物理脉冲信号。

核心特性:
    1. 直接脉冲编程: 使用 PulseProgram 创建脉冲序列
    2. 门电路到脉冲: Circuit → Pulse 自动分解
    3. TQASM 导出: 导出符合 OpenQASM 3.0 + OpenPulse 规范的代码
    4. 本地模拟: 使用物理真实的脉冲演化模拟器

参考文献:
    [1] OpenQASM 3.0: https://openqasm.com/
    [2] OpenPulse Grammar: https://openqasm.com/language/openpulse.html
    [3] QuTiP-qip Processor Model: Quantum 6, 630 (2022)

作者: TyxonQ Team
日期: 2025
"""

import numpy as np
from tyxonq import Circuit
from tyxonq.core.ir.pulse import PulseProgram
from tyxonq import waveforms


def example_1_direct_pulse_programming():
    """
    示例 1: 直接脉冲编程
    
    使用 PulseProgram 直接定义脉冲序列，实现单量子比特旋转。
    这是最底层的控制方式，适用于脉冲优化、最优控制等高级应用。
    """
    print("=" * 70)
    print("示例 1: 直接脉冲编程 (PulseProgram)")
    print("=" * 70)
    
    # 创建 1 量子比特的脉冲程序
    prog = PulseProgram(1)
    
    # 物理参数
    qubit_freq = 5.0e9  # 量子比特频率 5 GHz (典型超导量子比特)
    anharmonicity = -330e6  # 非谐性 -330 MHz
    
    # 添加 DRAG 脉冲 (Derivative Removal by Adiabatic Gate)
    # DRAG 脉冲可以抑制非谐性导致的泄漏误差
    prog.drag(
        qubit=0,
        amp=1.0,          # 幅度 (归一化到 π 旋转)
        duration=160,     # 持续时间 160 ns (典型值)
        sigma=40,         # 高斯宽度 40 ns
        beta=0.2,         # DRAG 系数
        qubit_freq=qubit_freq
    )
    
    print(f"\n脉冲程序:")
    print(f"  - 量子比特数: {prog.num_qubits}")
    print(f"  - 量子比特频率: {qubit_freq/1e9:.1f} GHz")
    print(f"  - 非谐性: {anharmonicity/1e6:.0f} MHz")
    
    # 方式 1: 直接数值模拟 (路径 B - 快速验证)
    print(f"\n方式 1: 直接数值模拟")
    state = prog.state(backend="numpy")
    print(f"  量子态: {state}")
    print(f"  归一化: {np.linalg.norm(state):.6f}")
    
    # 方式 2: 编译为 TQASM 并导出 (路径 A - 云端提交)
    print(f"\n方式 2: 编译为 TQASM")
    tqasm_code = prog.compile(
        output="tqasm",
        device_params={
            "qubit_freq": [qubit_freq],
            "anharmonicity": [anharmonicity]
        }
    )
    print(f"  TQASM 代码长度: {len(tqasm_code)} 字符")
    print(f"\n  TQASM 代码预览:")
    print("  " + "-" * 66)
    for line in tqasm_code.split('\n')[:10]:
        print(f"  {line}")
    print("  ...")
    
    return prog, state


def example_2_gate_to_pulse_conversion():
    """
    示例 2: 门电路到脉冲的自动转换
    
    从高层的门电路开始，自动分解为物理层的脉冲序列。
    这是最常用的方式，结合了门电路的便利性和脉冲级的精确控制。
    """
    print("\n" + "=" * 70)
    print("示例 2: 门电路 → 脉冲自动转换")
    print("=" * 70)
    
    # 创建标准门电路 (Bell 态)
    circuit = Circuit(2)
    circuit.h(0)      # Hadamard 门
    circuit.cx(0, 1)  # CNOT 门
    
    print(f"\n原始门电路:")
    print(f"  - 量子比特数: {circuit.num_qubits}")
    print(f"  - 门操作: H(0) · CX(0,1)")
    
    # 启用脉冲模式并提供设备参数
    circuit_pulse = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],          # 两个量子比特频率
        "anharmonicity": [-330e6, -320e6],     # 非谐性
        "coupling_strength": 5e6,              # 耦合强度 5 MHz
        "cx_duration": 400,                    # CX 门时间 400 ns
        "cr_amplitude": 0.3,                   # Cross-resonance 幅度
        "cr_echo": True                        # 启用 echo 脉冲抑制误差
    })
    
    # 编译为脉冲 IR
    from tyxonq.compiler.api import compile
    result = compile(circuit_pulse, output="pulse_ir", options={"mode": "pulse_only"})
    pulse_circuit = result["circuit"]
    
    print(f"\n编译后的脉冲电路:")
    print(f"  - 脉冲操作数: {len([op for op in pulse_circuit.ops if op[0] == 'pulse'])}")
    print(f"  - 脉冲库大小: {len(pulse_circuit.metadata.get('pulse_library', {}))}")
    
    # 分析脉冲分解
    print(f"\n脉冲分解分析:")
    h_pulses = sum(1 for op in pulse_circuit.ops if 'h_' in str(op))
    cx_pulses = sum(1 for op in pulse_circuit.ops if 'cx_' in str(op))
    print(f"  - H 门 → {h_pulses} 个脉冲 (RY(π/2) + RX(π))")
    print(f"  - CX 门 → {cx_pulses} 个脉冲 (pre-rotation + CR + echo + post-rotation)")
    
    # 物理时间估算
    h_time = 160  # ns
    cx_time = 400  # ns
    total_time = h_time + cx_time
    print(f"\n物理执行时间:")
    print(f"  - H 门: {h_time} ns")
    print(f"  - CX 门: {cx_time} ns")
    print(f"  - 总计: {total_time} ns = {total_time/1e3:.2f} μs")
    
    # 验证量子态
    state_gate = circuit.state(backend="numpy")
    state_pulse = circuit_pulse.state(backend="numpy")
    fidelity = abs(np.vdot(state_gate, state_pulse)) ** 2
    
    print(f"\n量子态验证:")
    print(f"  - 门级态: {state_gate}")
    print(f"  - 脉冲态: {state_pulse}")
    print(f"  - 保真度: {fidelity:.6f}")
    
    return circuit_pulse, pulse_circuit


def example_3_tqasm_export_with_defcal():
    """
    示例 3: 完整 TQASM + defcal 导出
    
    导出符合 OpenQASM 3.0 + OpenPulse 规范的 TQASM 代码，
    包含完整的 defcal 定义，可直接提交到支持 OpenPulse 的量子硬件。
    """
    print("\n" + "=" * 70)
    print("示例 3: TQASM + Defcal 导出 (OpenQASM 3.0)")
    print("=" * 70)
    
    # 创建简单电路
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    # 启用脉冲模式
    circuit_pulse = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6],
        "coupling_strength": 5e6,
        "cx_duration": 400
    })
    
    # 编译为 TQASM (包含完整 defcal 定义)
    from tyxonq.compiler.api import compile
    result = compile(
        circuit_pulse, 
        output="tqasm",
        options={"mode": "pulse_only"}
    )
    tqasm_code = result["circuit"]
    
    print(f"\n生成的 TQASM 代码:")
    print("=" * 70)
    print(tqasm_code)
    print("=" * 70)
    
    # 语法验证
    print(f"\n语法验证:")
    checks = [
        ("OpenQASM 3.0 版本", "OPENQASM 3.0" in tqasm_code),
        ("OpenPulse 语法", 'defcalgrammar "openpulse"' in tqasm_code),
        ("Cal 校准块", "cal {" in tqasm_code),
        ("Port 声明", "extern port" in tqasm_code),
        ("Frame 声明", "newframe(" in tqasm_code),
        ("Defcal 定义", "defcal" in tqasm_code),
        ("Waveform 定义", "waveform" in tqasm_code),
        ("Play 指令", "play(" in tqasm_code),
        ("物理量子比特", "$0" in tqasm_code),
    ]
    
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print(f"\n下一步: 可将 TQASM 代码提交到云端量子硬件执行")
    print(f"  示例: submit_to_tyxonq_cloud(tqasm_code)")
    
    return tqasm_code


def example_4_custom_pulse_waveforms():
    """
    示例 4: 自定义脉冲波形
    
    使用不同的脉冲波形 (DRAG, Gaussian, Constant) 实现精确控制。
    展示脉冲编程的灵活性和对物理层的完全掌控。
    """
    print("\n" + "=" * 70)
    print("示例 4: 自定义脉冲波形")
    print("=" * 70)
    
    prog = PulseProgram(1)
    qubit_freq = 5.0e9
    
    # 1. DRAG 脉冲 (抑制泄漏)
    print(f"\n1️⃣  DRAG 脉冲 (Derivative Removal by Adiabatic Gate):")
    print(f"   用途: X/Y 旋转，抑制非谐性导致的泄漏")
    prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=qubit_freq)
    
    # 2. Gaussian 脉冲 (平滑控制)
    print(f"\n2️⃣  Gaussian 脉冲:")
    print(f"   用途: 平滑脉冲包络，减少频谱泄漏")
    prog.gaussian(0, amp=0.5, duration=200, sigma=50, qubit_freq=qubit_freq)
    
    # 3. Constant 脉冲 (方波)
    print(f"\n3️⃣  Constant 脉冲 (方波):")
    print(f"   用途: 简单控制，测试用")
    prog.constant(0, amp=0.3, duration=100, qubit_freq=qubit_freq)
    
    # 4. Sine 脉冲 (连续波)
    print(f"\n4️⃣  Sine 脉冲 (连续波):")
    print(f"   用途: 共振驱动")
    prog.sine(0, amp=0.4, frequency=qubit_freq, duration=150, qubit_freq=qubit_freq)
    
    print(f"\n脉冲序列总结:")
    print(f"  - 总时长估算: ~{160+200+100+150} ns = {(160+200+100+150)/1e3:.1f} μs")
    
    # 数值模拟
    state = prog.state(backend="numpy")
    print(f"\n量子态:")
    print(f"  |0⟩ 分量: {abs(state[0]):.4f}")
    print(f"  |1⟩ 分量: {abs(state[1]):.4f}")
    print(f"  归一化: {np.linalg.norm(state):.6f}")
    
    return prog


def main():
    """主函数: 运行所有示例"""
    print("\n" + "=" * 70)
    print("TyxonQ 脉冲级量子编程完整示例")
    print("=" * 70)
    print("""
本示例展示 TyxonQ 的脉冲编程核心特性:

1. 直接脉冲编程 (PulseProgram)
2. 门电路 → 脉冲自动转换
3. TQASM + Defcal 导出 (OpenQASM 3.0)
4. 自定义脉冲波形

执行路径:
  路径 A: PulseProgram → compile(output="tqasm") → 云端硬件
  路径 B: PulseProgram → .state(backend="numpy") → 本地模拟
    """)
    
    # 运行所有示例
    prog1, state1 = example_1_direct_pulse_programming()
    circuit_pulse2, pulse_circuit2 = example_2_gate_to_pulse_conversion()
    tqasm_code3 = example_3_tqasm_export_with_defcal()
    prog4 = example_4_custom_pulse_waveforms()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有示例运行完成!")
    print("=" * 70)
    print("""
关键要点:

1. TyxonQ 支持两种脉冲编程方式:
   - 直接脉冲: PulseProgram (底层控制)
   - 门转脉冲: Circuit.use_pulse() (高层便利)

2. 两条独立执行路径:
   - 路径 A: 编译为 TQASM → 云端硬件 (生产环境)
   - 路径 B: 直接数值模拟 (开发调试)

3. 完全符合 OpenQASM 3.0 + OpenPulse 规范
   - 可直接提交到支持 OpenPulse 的量子硬件
   - 完整的 defcal 定义
   - 物理真实的脉冲参数

下一步:
  - 脉冲级 VQE 优化 (examples/pulse_vqe_optimization.py)
  - 最优控制 (examples/optimal_control_pulse.py)
  - 动态解耦 (examples/dynamical_decoupling.py)
    """)


if __name__ == "__main__":
    main()
