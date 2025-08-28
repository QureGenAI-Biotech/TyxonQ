"""
TyxonQ Circuit Chain API 全面演示 - 重构升级版

展示功能:
1. 基础量子电路构建与执行
2. 链式配置API (compile/device/postprocessing/run)
3. 自动补全机制
4. 全局默认配置
5. 多后端支持 (numpy/pytorch)
6. 高级量子门操作
7. 电路组合与重映射
8. JSON序列化
9. 硬件任务提交
10. 脉冲调度支持

运行此文件查看完整演示结果。
"""

from __future__ import annotations

import tyxonq as tq
from tyxonq import Circuit, Hamiltonian
import json
import numpy as np


def demo_basic_circuit_construction():
    """演示基础量子电路构建"""
    print("\n[Demo 1] 基础量子电路构建")
    
    # 方法1: 基础构建
    c1 = Circuit(2).h(0).cx(0, 1)
    print("电路1 - 基础构建:", c1.gate_summary())
    
    # 方法2: 使用大写别名
    c2 = Circuit(2).H(0).CX(0, 1).RZ(1, 0.5)
    print("电路2 - 大写别名:", c2.gate_summary())
    
    # 方法3: 带参数的量子门
    c3 = Circuit(3).h(0).rx(1, np.pi/4).rz(2, 1.2).cnot(0, 2)
    print("电路3 - 带参数门:", c3.gate_summary())
    
    return c1, c2, c3


def demo_chainable_api():
    """演示链式配置API"""
    print("\n[Demo 2] 链式配置API")
    
    # 完整链式调用
    result = (
        Circuit(2).h(0).cx(0, 1)
        .compile(compile_engine="qiskit")
        .device(provider="simulator", device="statevector", shots=1024)
        .postprocessing(method="none")
        .run()
    )
    print("完整链式调用结果:", result)
    
    # 部分链式调用 (自动补全)
    result2 = (
        Circuit(2).h(0).cx(0, 1)
        .compile()
        .run()  # device和postprocessing自动补全
    )
    print("部分链式调用结果:", result2)
    
    return result, result2


def demo_automatic_completion():
    """演示自动补全机制"""
    print("\n[Demo 3] 自动补全机制")
    
    # 只设置compile选项
    result1 = (
        Circuit(2).h(0).cx(0, 1)
        .compile()
        .run()  # device和postprocessing自动补全
    )
    
    # 只设置device选项
    result2 = (
        Circuit(2).h(0).cx(0, 1)
        .device(shots=512)
        .run()  # compile和postprocessing自动补全
    )
    
    # 只设置postprocessing选项
    result3 = (
        Circuit(2).h(0).cx(0, 1)
        .postprocessing(method="none")
        .run()  # compile和device自动补全
    )
    
    print("仅compile设置结果:", result1)
    print("仅device设置结果:", result2)
    print("仅postprocessing设置结果:", result3)
    
    return result1, result2, result3


def demo_global_defaults():
    """演示全局默认配置"""
    print("\n[Demo 4] 全局默认配置")
    
    # 设置全局默认值
    tq.device(provider="simulator", device="statevector", shots=2048)
    tq.postprocessing(method="none")
    tq.compile(compile_engine="default")
    
    # 使用全局默认值执行
    result = Circuit(2).h(0).cx(0, 1).run()
    print("全局默认配置结果:", result)
    
    # 重置全局默认值
    tq.device(provider="simulator", device="statevector", shots=1024)
    
    return result


def demo_multiple_backends():
    """演示多后端支持"""
    print("\n[Demo 5] 多后端支持")
    
    # 设置numpy后端
    tq.set_backend("numpy")
    result_numpy = Circuit(2).h(0).cx(0, 1).run()
    print("NumPy后端结果:", result_numpy)
    
    # 尝试设置pytorch后端 (如果可用)
    try:
        tq.set_backend("pytorch")
        result_torch = Circuit(2).h(0).cx(0, 1).run()
        print("PyTorch后端结果:", result_torch)
    except Exception as e:
        print("PyTorch后端不可用:", str(e))
        result_torch = None
    
    return result_numpy, result_torch


def demo_advanced_gates():
    """演示高级量子门操作"""
    print("\n[Demo 6] 高级量子门操作")
    
    # 测量操作
    c1 = Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
    print("带测量操作:", c1.gate_summary())
    
    # 重置操作
    c2 = Circuit(2).h(0).reset(0).h(0)
    print("带重置操作:", c2.gate_summary())
    
    # 屏障操作
    c3 = Circuit(3).h(0).cx(0, 1).add_barrier().h(2).cx(1, 2)
    print("带屏障操作:", c3.gate_summary())
    
    # 指令式测量
    c4 = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    print("指令式测量:", c4.instructions)
    
    return c1, c2, c3, c4


def demo_circuit_composition():
    """演示电路组合与重映射"""
    print("\n[Demo 7] 电路组合与重映射")
    
    # 创建基础电路
    bell = Circuit(2).h(0).cx(0, 1)
    rotation = Circuit(1).rz(0, np.pi/4)
    
    # 电路组合 - 将rotation电路的量子位0映射到bell电路的量子位1
    combined = bell.compose(rotation, indices=[1])
    print("电路组合结果:", combined.gate_summary())
    
    # 量子位重映射
    remapped = bell.remap_qubits({0: 2, 1: 3}, new_num_qubits=4)
    print("量子位重映射:", remapped.gate_summary())
    
    # 电路扩展
    extended = bell.extended([("h", 0), ("cx", 1, 0)])
    print("电路扩展:", extended.gate_summary())
    
    return combined, remapped, extended


def demo_json_serialization():
    """演示JSON序列化"""
    print("\n[Demo 8] JSON序列化")
    
    # 创建电路
    circuit = Circuit(2).h(0).cx(0, 1).with_metadata(description="Bell state circuit")
    
    # 序列化为JSON对象
    json_obj = circuit.to_json_obj()
    print("JSON对象:", json.dumps(json_obj, indent=2))
    
    # 序列化为JSON字符串
    json_str = circuit.to_json_str(indent=2)
    print("JSON字符串长度:", len(json_str))
    
    # 从JSON反序列化
    reconstructed = Circuit.from_json_str(json_str)
    print("反序列化电路:", reconstructed.gate_summary())
    
    return circuit, json_str, reconstructed


def demo_qasm_compilation():
    """演示QASM编译"""
    print("\n[Demo 9] QASM编译")
    
    # 编译为OpenQASM 2.0
    circuit = Circuit(2).h(0).cx(0, 1).compile()
    qasm2 = circuit.compile(output="qasm2", compile_engine="qiskit")
    print("OpenQASM 2.0:\n", qasm2)
    
    # 编译为OpenQASM 3.0 (如果支持)
    try:
        qasm3 = circuit.compile(output="qasm3", compile_engine="qiskit")
        print("OpenQASM 3.0:\n", qasm3)
    except Exception as e:
        print("OpenQASM 3.0编译失败:", str(e))
        qasm3 = None
    
    return qasm2, qasm3


def demo_hamiltonian():
    """演示Hamiltonian使用"""
    print("\n[Demo 10] Hamiltonian使用")
    
    # 创建Hamiltonian
    hamiltonian = Hamiltonian(terms=[("Z", 0), ("X", 1), ("Y", 2)])
    print("Hamiltonian创建成功:", hamiltonian)
    
    return hamiltonian


def demo_hardware_submission():
    """演示硬件任务提交"""
    print("\n[Demo 11] 硬件任务提交 (模拟)")
    
    # 创建电路
    circuit = Circuit(2).h(0).cx(0, 1)
    
    # 模拟硬件提交
    try:
        result = circuit.run(provider="simulator", device="statevector", shots=100)
        print("硬件模拟结果:", result)
    except Exception as e:
        print("硬件提交失败:", str(e))
        result = None
    
    return result


def demo_pulse_support():
    """演示脉冲调度支持"""
    print("\n[Demo 12] 脉冲调度支持")
    
    try:
        from tyxonq import PulseInstruction, PulseSchedule
        
        # 创建脉冲指令
        pulse = PulseInstruction("gaussian", qubits=(0,), parameters={"duration": 100, "sigma": 25})
        schedule = PulseSchedule(instructions=[pulse])
        
        print("脉冲指令:", pulse)
        print("脉冲调度:", schedule)
        
        return pulse, schedule
    except ImportError:
        print("脉冲模块不可用")
        return None, None


if __name__ == "__main__":
    """运行所有演示"""
    print("=" * 60)
    print("TyxonQ Circuit Chain API 全面演示")
    print("=" * 60)
    
    # 运行所有演示
    demo_basic_circuit_construction()
    demo_chainable_api()
    demo_automatic_completion()
    demo_global_defaults()
    demo_multiple_backends()
    demo_advanced_gates()
    demo_circuit_composition()
    demo_json_serialization()
    demo_qasm_compilation()
    demo_hamiltonian()
    demo_hardware_submission()
    demo_pulse_support()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


def demo_global_defaults():
    print("\n[Demo 4] Global defaults: tyxonq.device / tyxonq.postprocessing")
    # Set global defaults once
    tq.device(provider="simulator", device="statevector", shots=256)
    tq.postprocessing(method="none")

    c = Circuit(2).H(0).CX(0, 1)
    result = c.run()
    print("Result (dict):", result)
    return result


def demo_qasm_compile_view():
    print("\n[Demo 5] Compile to OpenQASM for inspection (does not submit)")
    c = Circuit(2).H(0).CX(0, 1).compile()
    qasm = c.compile(output="qasm2", compile_engine="qiskit")
    print("QASM:\n", qasm)
    return qasm


if __name__ == "__main__":
    # Run demos sequentially and print their outputs
    demo_automatic_completion()
    demo_global_defaults()
    demo_qasm_compile_view()


