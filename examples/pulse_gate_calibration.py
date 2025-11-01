#!/usr/bin/env python3
"""
脉冲级门校准与优化示例

本示例展示如何使用 TyxonQ 的脉冲编程功能实现:
    1. 自定义门校准 (Custom Gate Calibration)
    2. 脉冲参数优化 (Pulse Parameter Optimization)
    3. Cross-Resonance 门调优
    4. 门保真度分析

应用场景:
    - 量子硬件校准
    - 门误差抑制
    - 最优控制
    - VQE 等变分算法的脉冲级优化

参考文献:
    [1] McKay et al., PRA 96, 022330 (2017) - Efficient Z gates
    [2] IBM Qiskit Pulse: Pulse-level programming
    [3] Rigetti: arXiv:1903.02492 - Parametric gates

作者: TyxonQ Team
日期: 2025
"""

import numpy as np
from tyxonq import Circuit
from tyxonq.core.ir.pulse import PulseProgram
from tyxonq import waveforms
from tyxonq.compiler.api import compile


def example_1_custom_x_gate_calibration():
    """
    示例 1: 自定义 X 门校准
    
    通过调整 DRAG 脉冲参数来优化 X 门的实现。
    展示如何通过脉冲级控制提升门保真度。
    """
    print("=" * 70)
    print("示例 1: 自定义 X 门校准")
    print("=" * 70)
    
    qubit_freq = 5.0e9
    anharmonicity = -330e6
    
    # 方案 1: 标准 X 门 (π 旋转)
    print("\n方案 1: 标准 X 门 (DRAG 脉冲)")
    prog_std = PulseProgram(1)
    prog_std.drag(
        qubit=0,
        amp=1.0,          # π 旋转
        duration=160,     # 160 ns
        sigma=40,
        beta=0.2,         # 标准 DRAG 系数
        qubit_freq=qubit_freq
    )
    state_std = prog_std.state(backend="numpy")
    print(f"  目标态: |1⟩")
    print(f"  实际态: {state_std}")
    print(f"  |1⟩ 分量: {abs(state_std[1]):.6f}")
    
    # 方案 2: 优化 X 门 (调整 beta 抑制泄漏)
    print("\n方案 2: 优化 X 门 (调整 DRAG beta)")
    prog_opt = PulseProgram(1)
    prog_opt.drag(
        qubit=0,
        amp=1.0,
        duration=160,
        sigma=40,
        beta=0.4,         # 更大的 beta 抑制泄漏到 |2⟩
        qubit_freq=qubit_freq
    )
    state_opt = prog_opt.state(backend="numpy")
    print(f"  目标态: |1⟩")
    print(f"  实际态: {state_opt}")
    print(f"  |1⟩ 分量: {abs(state_opt[1]):.6f}")
    
    # 保真度对比
    target_state = np.array([0, 1])  # |1⟩
    fidelity_std = abs(np.vdot(target_state, state_std)) ** 2
    fidelity_opt = abs(np.vdot(target_state, state_opt)) ** 2
    
    print(f"\n保真度对比:")
    print(f"  标准 X 门: {fidelity_std:.6f}")
    print(f"  优化 X 门: {fidelity_opt:.6f}")
    print(f"  改进: {(fidelity_opt - fidelity_std)*100:.4f}%")
    
    return prog_std, prog_opt


def example_2_virtual_z_gate():
    """
    示例 2: Virtual-Z 门 (零时间门)
    
    Z 旋转可以通过相位帧更新实现,不需要物理脉冲,节省门时间。
    这是现代超导量子处理器的关键优化技术。
    """
    print("\n" + "=" * 70)
    print("示例 2: Virtual-Z 门 (Frame Update)")
    print("=" * 70)
    
    # 创建包含 RZ 门的电路
    circuit = Circuit(1)
    circuit.h(0)       # Hadamard
    circuit.rz(np.pi/4, 0)  # RZ(π/4) - 将被编译为 Virtual-Z
    circuit.h(0)       # Hadamard
    
    print(f"\n电路结构:")
    print(f"  H(0) → RZ(π/4, 0) → H(0)")
    
    # 编译为脉冲
    circuit_pulse = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9],
        "anharmonicity": [-330e6]
    })
    
    result = compile(circuit_pulse, output="pulse_ir", options={"mode": "pulse_only"})
    pulse_circuit = result["circuit"]
    
    # 分析脉冲序列
    print(f"\n脉冲序列分析:")
    virtual_z_count = sum(1 for op in pulse_circuit.ops if op[0] == "virtual_z")
    pulse_count = sum(1 for op in pulse_circuit.ops if op[0] == "pulse")
    
    print(f"  - Virtual-Z 操作: {virtual_z_count} 个 (零时间!)")
    print(f"  - 物理脉冲: {pulse_count} 个 (H 门分解)")
    
    # 时间估算
    h_time = 160 * 2  # 两个 H 门
    virtual_z_time = 0  # Virtual-Z 零时间
    
    print(f"\n门时间对比:")
    print(f"  - 传统 RZ (物理脉冲): ~160 ns")
    print(f"  - Virtual-Z (帧更新): 0 ns ✨")
    print(f"  - 总电路时间: {h_time} ns (仅 H 门)")
    
    # 导出 TQASM 查看 Virtual-Z 实现
    tqasm_result = compile(circuit_pulse, output="tqasm", options={"mode": "pulse_only"})
    tqasm_code = tqasm_result["circuit"]
    
    print(f"\n TQASM 中的 Virtual-Z 实现:")
    print(f"  (在 defcal 中表现为 shift_phase 指令)")
    
    if "shift_phase" in tqasm_code:
        print(f"  ✅ 找到 shift_phase 指令 (Virtual-Z)")
    
    return circuit_pulse


def example_3_cx_gate_optimization():
    """
    示例 3: CX 门参数优化
    
    通过调整 Cross-Resonance 脉冲参数来优化 CX 门。
    这是双量子比特门优化的核心技术。
    """
    print("\n" + "=" * 70)
    print("示例 3: CX 门 Cross-Resonance 优化")
    print("=" * 70)
    
    # 创建 Bell 态电路
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    print(f"\n电路: Bell 态制备")
    print(f"  H(0) → CX(0, 1)")
    
    # 方案 1: 标准 CX 参数
    print(f"\n方案 1: 标准 CX 参数")
    circuit_std = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6],
        "coupling_strength": 5e6,      # 5 MHz 耦合
        "cx_duration": 400,            # 400 ns
        "cr_amplitude": 0.3,           # CR 幅度
        "cr_echo": True                # Echo 脉冲
    })
    
    state_std = circuit_std.state(backend="numpy")
    
    # 方案 2: 优化 CX (更强的 CR 幅度,更短的门时间)
    print(f"\n方案 2: 优化 CX (更强 CR 幅度)")
    circuit_opt = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6],
        "coupling_strength": 8e6,      # 8 MHz 耦合 (更强)
        "cx_duration": 300,            # 300 ns (更快)
        "cr_amplitude": 0.5,           # 更大的 CR 幅度
        "cr_echo": True
    })
    
    state_opt = circuit_opt.state(backend="numpy")
    
    # Bell 态保真度
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |Φ+⟩
    fidelity_std = abs(np.vdot(bell_state, state_std)) ** 2
    fidelity_opt = abs(np.vdot(bell_state, state_opt)) ** 2
    
    print(f"\nBell 态保真度:")
    print(f"  标准参数: {fidelity_std:.6f}")
    print(f"  优化参数: {fidelity_opt:.6f}")
    
    # 门时间对比
    print(f"\n门时间对比:")
    print(f"  标准 CX: 400 ns")
    print(f"  优化 CX: 300 ns (节省 25%)")
    
    return circuit_std, circuit_opt


def example_4_pulse_parameter_sweep():
    """
    示例 4: 脉冲参数扫描
    
    通过扫描脉冲参数 (幅度、持续时间等) 来寻找最优设置。
    这是脉冲级校准的标准方法。
    """
    print("\n" + "=" * 70)
    print("示例 4: 脉冲参数扫描 (幅度优化)")
    print("=" * 70)
    
    qubit_freq = 5.0e9
    
    # 扫描 DRAG 脉冲幅度
    amplitudes = np.linspace(0.5, 1.5, 11)  # 从 0.5 到 1.5
    fidelities = []
    
    print(f"\n扫描 DRAG 脉冲幅度 (目标: X 门, |0⟩ → |1⟩)")
    print(f"  幅度范围: {amplitudes[0]:.1f} 到 {amplitudes[-1]:.1f}")
    print(f"  扫描点数: {len(amplitudes)}")
    
    target_state = np.array([0, 1])  # |1⟩
    
    for amp in amplitudes:
        prog = PulseProgram(1)
        prog.drag(
            qubit=0,
            amp=amp,
            duration=160,
            sigma=40,
            beta=0.2,
            qubit_freq=qubit_freq
        )
        state = prog.state(backend="numpy")
        fidelity = abs(np.vdot(target_state, state)) ** 2
        fidelities.append(fidelity)
    
    # 找到最优幅度
    best_idx = np.argmax(fidelities)
    best_amp = amplitudes[best_idx]
    best_fidelity = fidelities[best_idx]
    
    print(f"\n扫描结果:")
    print(f"  最优幅度: {best_amp:.3f}")
    print(f"  最高保真度: {best_fidelity:.6f}")
    
    # 显示前5个最好的结果
    print(f"\n保真度 Top 5:")
    sorted_indices = np.argsort(fidelities)[::-1][:5]
    for i, idx in enumerate(sorted_indices, 1):
        print(f"    {i}. 幅度 {amplitudes[idx]:.3f} → 保真度 {fidelities[idx]:.6f}")
    
    return amplitudes, fidelities, best_amp


def example_5_gate_calibration_export():
    """
    示例 5: 门校准导出为 TQASM defcal
    
    将优化后的门校准导出为 TQASM defcal 定义,
    可以作为硬件校准文件使用。
    """
    print("\n" + "=" * 70)
    print("示例 5: 门校准导出为 TQASM Defcal")
    print("=" * 70)
    
    # 创建优化后的 X 和 CX 门
    circuit = Circuit(2)
    circuit.x(0)       # 优化的 X 门
    circuit.cx(0, 1)   # 优化的 CX 门
    
    # 使用优化参数
    circuit_pulse = circuit.use_pulse(device_params={
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6],
        "coupling_strength": 8e6,      # 优化后的耦合强度
        "cx_duration": 300,            # 优化后的 CX 时间
        "cr_amplitude": 0.5,           # 优化后的 CR 幅度
        "cr_echo": True
    })
    
    # 导出为 TQASM (包含优化后的 defcal)
    result = compile(circuit_pulse, output="tqasm", options={"mode": "pulse_only"})
    tqasm_code = result["circuit"]
    
    print(f"\n导出的 TQASM 校准文件:")
    print("=" * 70)
    print(tqasm_code)
    print("=" * 70)
    
    print(f"\n校准文件特性:")
    print(f"  - 包含优化后的 defcal x $0")
    print(f"  - 包含优化后的 defcal cx $0, $1")
    print(f"  - 所有脉冲参数已嵌入")
    print(f"  - 可直接用作硬件校准文件")
    
    # 保存到文件
    calibration_file = "optimized_gate_calibration.qasm"
    print(f"\n可保存为: {calibration_file}")
    print(f"  用法: include \"{calibration_file}\";")
    
    return tqasm_code


def main():
    """主函数: 运行所有门校准示例"""
    print("\n" + "=" * 70)
    print("TyxonQ 脉冲级门校准与优化示例")
    print("=" * 70)
    print("""
本示例展示脉冲级门校准的完整流程:

1. 自定义 X 门校准 (DRAG 参数优化)
2. Virtual-Z 门 (零时间 Z 旋转)
3. CX 门 Cross-Resonance 优化
4. 脉冲参数扫描 (自动校准)
5. 校准导出为 TQASM defcal

关键技术:
  - DRAG 脉冲抑制泄漏
  - Virtual-Z 节省门时间
  - Cross-Resonance 双比特门
  - 参数优化提升保真度
    """)
    
    # 运行所有示例
    prog_std1, prog_opt1 = example_1_custom_x_gate_calibration()
    circuit_vz2 = example_2_virtual_z_gate()
    circuit_std3, circuit_opt3 = example_3_cx_gate_optimization()
    amps4, fids4, best_amp4 = example_4_pulse_parameter_sweep()
    tqasm_code5 = example_5_gate_calibration_export()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有门校准示例运行完成!")
    print("=" * 70)
    print("""
关键收获:

1. 脉冲级控制提供更高精度
   - 可调整幅度、持续时间、相位等所有物理参数
   - 通过 DRAG 系数 beta 抑制泄漏到高能级

2. Virtual-Z 门节省时间
   - Z 旋转通过相位帧更新实现
   - 零物理时间,只更新软件状态

3. CX 门可通过 Cross-Resonance 优化
   - 调整 CR 幅度和耦合强度
   - 在保真度和速度之间权衡

4. 参数扫描找到最优设置
   - 自动化校准流程
   - 数据驱动的门优化

5. 校准结果可导出为 TQASM defcal
   - 符合 OpenQASM 3.0 规范
   - 可直接用于硬件

下一步:
  - 结合 VQE 的脉冲级优化
  - 动态解耦序列设计
  - 最优控制脉冲合成
    """)


if __name__ == "__main__":
    main()
