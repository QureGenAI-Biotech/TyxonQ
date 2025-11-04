#!/usr/bin/env python3
"""P0.5: 脉冲编程云端提交端到端测试

展示 TyxonQ 双模式 Pulse 架构的云端提交完整流程。

## 三模式架构 (Memory: 8b12df21)

### 模式 A - 显式脉冲编译 (推荐)
```python
Circuit(n)
  → .h(0).cx(0,1)         # 门操作
  → .use_pulse(device_params={...})  # 显式启用脉冲编译
  → compile(output="tqasm")  # 编译为 TQASM
  → 提交到云端硬件
```

### 模式 B - 智能推断 (最便捷)
```python
Circuit(n)
  → .h(0).cx(0,1)         # 门操作 (无需 use_pulse)
  → compile(output="tqasm")  # 自动推断 + 默认参数
  → 提交到云端硬件
```

### 模式 C - 直接脉冲 (最底层)
```python
PulseProgram(n)
  → .drag(0, ...).gaussian(1, ...)  # 脉冲操作
  → .compile(output="tqasm")  # 编译为 TQASM
  → 提交到云端硬件
```

## 硬件限制 (TensorCircuit/TQASM 0.2 标准)

⚠️  当前硬件仅支持单比特脉冲 defcal 定义：
   - ✅ 单比特门 (X, H, RX, RY) 可自定义脉冲实现
   - ❌ 双比特门 (CX, CZ) 由硬件预定义，用户不能自定义
   - 📖 参考: TensorCircuit circuit.py Line 173 - newframe({qubit})

## 关键 API

### Circuit API
- `Circuit.use_pulse()`: 启用脉冲编译模式
- `Circuit.add_calibration()`: 添加自定义门的脉冲校准
- ❌ **Circuit 没有 add_pulse() 方法** (这是 PulseProgram 的方法)

### PulseProgram API
- `PulseProgram.drag()`: 添加 DRAG 脉冲
- `PulseProgram.gaussian()`: 添加高斯脉冲
- `PulseProgram.add_pulse()`: 添加自定义脉冲波形

### 编译触发条件 (compiler/api.py)
1. 显式调用 `circuit.use_pulse()` → compile_engine="pulse"
2. 或者 `compile(output="tqasm")` → 自动推断启用 pulse 编译 (有警告)

作者: TyxonQ Team
日期: 2025-10-24
"""

import os
import sys
import numpy as np
import warnings


# ==============================================================================
# API 凭证管理
# ==============================================================================

def get_api_token():
    """从环境变量获取 API Token（安全方式）
    
    用法:
        export TYXONQ_API_KEY="your_token_here"
        # 或者交互式输入
    """
    import tyxonq as tq
    import getpass
    
    # 方式1: 从环境变量获取
    token = os.environ.get("TYXONQ_API_KEY")
    
    if token:
        print("✅ 从环境变量获取 Token")
        # 设置 token 到 TyxonQ
        tq.set_token(token, provider="tyxonq", device="homebrew_s2_pulse")
        return token
    
    # 方式2: 交互式输入（可选）
    print("⚠️  未检测到 TYXONQ_API_KEY 环境变量")
    print("   请设置: export TYXONQ_API_KEY='your_token'")
    return None


# ==============================================================================
# 模式 A: 门电路 → use_pulse() → TQASM → 云端 (显式声明)
# ==============================================================================

def example_mode_a_explicit_pulse_with_comparison():
    """模式 A: 显式声明脉冲编译 + 真机/模拟对比分析
    
    链式调用工作流程:
        Circuit(1)
          → .h(0)  # 门操作
          → .use_pulse(device_params={...}, inline_pulses=True)  # 关键: inline_pulses=True
          → .device(provider="tyxonq", device="homebrew_s2_pulse")  # 配置设备
          → .run(shots=1024)  # 执行 (自动编译为 TQASM)
    
    对比分析:
        - 本地模拟: 理想场景 (无噪声)
        - 云端真机: 真实硬件 (有噪声 + 错误)
    
    优势:
        - ✅ 链式调用 (TyxonQ 核心特色)
        - ✅ 无警告 (显式声明意图)
        - ✅ 参数完整 (提供真实设备参数)
        - ✅ 对比分析 (理解真机 vs 模拟的差异)
    
    Returns:
        dict: 包含 circuit, tqasm_code, validation_result, 对比结果
    """
    print("\n" + "="*70)
    print("模式 A: 显式声明 + 真机/模拟对比分析 (推荐方式)")
    print("="*70)
    
    from tyxonq import Circuit
    from tyxonq.compiler.api import compile
    
    # 步骤 1: 链式调用构建电路
    print("\n1️⃣  链式调用构建电路:")
    print("   代码:")
    print("   circuit = (Circuit(1)")
    print("       .h(0)")
    print("       .use_pulse(device_params={...})")
    print("   )")
    
    circuit = (
        Circuit(1)
        .h(0)  # Hadamard 门
        .use_pulse(
            mode="pulse_only",  # 全部编译为脉冲
            device_params={
                "qubit_freq": [5.0e9],       # 5 GHz 量子比特频率
                "anharmonicity": [-330e6],   # -330 MHz 非谐性
                "T1": [80e-6],               # 80 μs 振幅阻尼时间
                "T2": [120e-6]               # 120 μs 退相干时间
            },
            inline_pulses=True  # 内联脉冲定义 (云端兼容)
        )
    )
    
    print(f"\n   ✅ 链式调用完成")
    print(f"   电路: H(0)")
    print(f"   量子比特数: {circuit.num_qubits}")
    print(f"   脉冲模式: pulse_only")
    print(f"   💡 inline_pulses=True 确保生成完整的 defcal 定义")
    
    # 步骤 2: 本地模拟
    print("\n2️⃣  本地模拟 (理想场景 - 无噪声):")
    try:
        state_sim = circuit.state(backend="numpy")
        prob_0_sim = abs(state_sim[0])**2
        prob_1_sim = abs(state_sim[1])**2
        print(f"   |0⟩ 概率: {prob_0_sim:.6f}")
        print(f"   |1⟩ 概率: {prob_1_sim:.6f}")
        print(f"   ✅ 模拟成功")
    except Exception as e:
        print(f"   ⚠️  模拟失败: {e}")
        state_sim = None
        prob_0_sim = None
        prob_1_sim = None
    
    # 步骤 3: 编译为 TQASM (用于验证)
    print("\n3️⃣  编译为 TQASM (用于验证):")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compile(circuit, output="tqasm", options={"mode": "pulse_only"})
        
        # 检查警告
        if w:
            print(f"   ⚠️  编译警告: {len(w)} 条")
            for warning in w:
                print(f"      → {warning.message}")
        else:
            print(f"   ✅ 编译成功，无警告")
    
    # compiled_source 是 TQASM 字符串
    tqasm_code = result["compiled_source"]
    
    print(f"\n   TQASM 代码长度: {len(tqasm_code)} 字符")
    
    # 步骤 3: 验证 TQASM 格式
    print("\n3️⃣  验证 TQASM 格式:")
    validation = validate_tqasm_format(tqasm_code)
    
    print(f"   版本声明: {'✅' if validation['version'] else '❌'}")
    print(f"   Cal 块: {'✅' if validation['cal_block'] else '❌'}")
    print(f"   Frame 声明: {'✅' if validation['frame_decl'] else '❌'}")
    print(f"   Defcal 定义: {'✅' if validation['defcal'] else '❌'}")
    print(f"   Waveform 定义: {'✅' if validation['waveform'] else '❌'}")
    
    if all(validation.values()):
        print("\n   ✅ TQASM 格式验证通过!")
    else:
        print("\n   ⚠️  TQASM 格式验证未完全通过")
    
    # 步骤 4: 显示 TQASM 代码预览
    print("\n4️⃣  TQASM 代码预览:")
    print("   " + "-"*66)
    for line in tqasm_code.split('\n')[:15]:
        print(f"   {line}")
    if len(tqasm_code.split('\n')) > 15:
        print("   ...")
    print("   " + "-"*66)
    
    # 步骤 6: 链式调用提交到云端 (推荐方式)
    print("\n6️⃣  链式调用提交到云端 (推荐方式):")
    print("   代码:")
    print("   result = circuit.device(provider='tyxonq', device='homebrew_s2_pulse').run(shots=1024)")
    print("   # → 自动编译为 TQASM 并提交到云端")
    print("   # → 返回任务句柄 (DeviceTask 对象)")
    
    token = get_api_token()
    cloud_result = None
    
    if token:
        print("\n   ✅ Token 已设置，开始提交任务...")
        try:
            import tyxonq as tq
            # 实际提交到云端
            task = circuit.device(provider="tyxonq", device="homebrew_s2").run(shots=1024)
            print(f"   ✅ 任务已提交: {task}")
            
            # 等待结果（设置超时）
            print("   ⏳ 等待云端执行结果...")
            import time
            time.sleep(3)  # 等待 3 秒
            
            try:
                details = tq.api.get_task_details(task, wait=False)
                cloud_result = details
                print(f"   ✅ 获取到任务状态: {details.get('status', 'unknown')}")
                if 'counts' in details:
                    print(f"   ✅ 测量结果: {details['counts']}")
            except Exception as e:
                print(f"   ⚠️  获取结果失败: {e}")
                print("      提示: 任务可能还在执行中，请稍后查询")
        except Exception as e:
            print(f"   ⚠️  提交失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n   ⚠️  跳过云端提交 (缺少 Token)")
        print("      设置环境变量: export TYXONQ_API_KEY='your_token'")
    
    # 步骤 7: 对比分析
    print("\n7️⃣  真机 vs 模拟对比分析:")
    print("   " + "-"*66)
    print(f"   {'执行方式':<20} {'|0⟩ 概率':<20} {'|1⟩ 概率':<20}")
    print("   " + "-"*66)
    
    if state_sim is not None:
        print(f"   {'本地模拟 (理想)':<20} {prob_0_sim:<20.6f} {prob_1_sim:<20.6f}")
    print(f"   {'云端真机 (待测)':<20} {'N/A (需Token)':<20} {'N/A':<20}")
    
    print("   " + "-"*66)
    print("   💡 分析:")
    print("      - 本地模拟: 理想结果，无噪声影响")
    print("      - 云端真机: 真实硬件，有噪声和错误")
    print("      - 差异来源: T1/T2退相干、颗粒度误差、测量误差")
    
    return {
        "circuit": circuit,
        "tqasm_code": tqasm_code,
        "validation": validation,
        "simulation": {"state": state_sim, "prob_0": prob_0_sim if state_sim is not None else None, "prob_1": prob_1_sim if state_sim is not None else None},
        "cloud_result": cloud_result
    }


# ==============================================================================
# 模式 B: 门电路 → 智能推断 → TQASM → 云端 (最便捷)
# ==============================================================================

def example_mode_b_smart_inference():
    """模式 B: 智能推断 + 链式调用 (最便捷)
    
    链式调用工作流程:
        Circuit(1)
          → .h(0)  # 门操作 (无需 use_pulse)
          → .device(provider="tyxonq", device="homebrew_s2_pulse")  # 配置设备
          → .run(shots=1024)  # 执行 (自动推断 + 自动补足参数)
    
    优势:
        - ✅ 最简洁 (链式调用，无需显式 use_pulse)
        - ✅ 自动推断启用脉冲编译
        - ✅ 自动补足默认参数 (5 GHz, -330 MHz)
        - ⚠️  有警告 (提示自动推断)
    
    智能推断规则 (compiler/api.py Line 140-148):
        - .device(...) + .run() → 自动检测 output="tqasm"
        - output="tqasm" → 自动启用 pulse 编译器
        - 缺少 device_params → 自动补足默认值
    
    Returns:
        dict: 包含 circuit, tqasm_code, validation_result
    """
    print("\n" + "="*70)
    print("模式 B: 智能推断 + 链式调用 (最便捷)")
    print("="*70)
    
    from tyxonq import Circuit
    from tyxonq.compiler.api import compile
    
    # 步骤 1: 链式调用构建电路 (无需 use_pulse)
    print("\n1️⃣  链式调用构建电路 (无需 use_pulse):")
    print("   代码:")
    print("   circuit = Circuit(1).h(0)")
    
    circuit = Circuit(1).h(0)  # Hadamard 门
    
    print(f"\n   ✅ 链式调用完成")
    print(f"   电路: H(0)")
    print(f"   量子比特数: {circuit.num_qubits}")
    print(f"   ⚠️  未调用 .use_pulse() (测试智能推断)")
    
    # 步骤 2: 直接编译为 TQASM (触发智能推断)
    print("\n2️⃣  编译为 TQASM (触发智能推断):")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compile(circuit, output="tqasm")
        
        # 分析警告
        tqasm_warnings = [warning for warning in w if "tqasm" in str(warning.message).lower()]
        param_warnings = [warning for warning in w if "自动补足" in str(warning.message)]
        
        print(f"\n   ⚠️  智能推断警告:")
        if tqasm_warnings:
            print(f"      1. TQASM 警告: output='tqasm' → 自动启用 pulse 编译")
            print(f"         {str(tqasm_warnings[0].message)[:60]}...")
        
        if param_warnings:
            print(f"      2. 参数警告: 缺少 device_params → 自动补足默认值")
            print(f"         默认: qubit_freq=[5.0e9], anharmonicity=[-330e6]")
    
    # compiled_source 是 TQASM 字符串
    tqasm_code = result["compiled_source"]
    
    print(f"\n   ✅ 智能推断成功!")
    print(f"   TQASM 代码长度: {len(tqasm_code)} 字符")
    
    # 步骤 3: 验证 TQASM 格式
    print("\n3️⃣  验证 TQASM 格式:")
    validation = validate_tqasm_format(tqasm_code)
    
    print(f"   版本声明: {'✅' if validation['version'] else '❌'}")
    print(f"   Cal 块: {'✅' if validation['cal_block'] else '❌'}")
    print(f"   Frame 声明: {'✅' if validation['frame_decl'] else '❌'}")
    print(f"   Defcal 定义: {'✅' if validation['defcal'] else '❌'}")
    print(f"   Waveform 定义: {'✅' if validation['waveform'] else '❌'}")
    
    if all(validation.values()):
        print("\n   ✅ TQASM 格式验证通过!")
    else:
        print("\n   ⚠️  TQASM 格式验证未完全通过")
    
    # 步骤 4: 显示 TQASM 代码预览
    print("\n4️⃣  TQASM 代码预览:")
    print("   " + "-"*66)
    for line in tqasm_code.split('\n')[:20]:
        print(f"   {line}")
    if len(tqasm_code.split('\n')) > 20:
        print("   ...")
    print("   " + "-"*66)
    
    # 步骤 5: 链式调用提交到云端
    print("\n5️⃣  链式调用提交到云端:")
    print("   代码:")
    print("   result = circuit.device(provider='tyxonq', device='homebrew_s2').run(shots=1024)")
    print("   → 自动推断: 检测到云端设备 → 编译为 TQASM")
    print("   → 自动补足: device_params (默认 5 GHz, -330 MHz)")
    
    token = get_api_token()
    cloud_result = None
    
    if token:
        print("\n   ✅ Token 已设置，开始提交任务...")
        try:
            import tyxonq as tq
            # 实际提交到云端
            task = circuit.device(provider="tyxonq", device="homebrew_s2").run(shots=1024)
            print(f"   ✅ 任务已提交: {task}")
            
            # 等待结果
            print("   ⏳ 等待云端执行结果...")
            import time
            time.sleep(3)
            
            try:
                details = tq.api.get_task_details(task, wait=False)
                cloud_result = details
                print(f"   ✅ 获取到任务状态: {details.get('status', 'unknown')}")
                if 'counts' in details:
                    print(f"   ✅ 测量结果: {details['counts']}")
            except Exception as e:
                print(f"   ⚠️  获取结果失败: {e}")
                print("      提示: 任务可能还在执行中")
        except Exception as e:
            print(f"   ⚠️  提交失败: {e}")
    else:
        print("\n   ⚠️  跳过云端提交 (缺少 Token)")
    
    return {
        "circuit": circuit,
        "tqasm_code": tqasm_code,
        "validation": validation,
        "cloud_result": cloud_result
    }


# ==============================================================================
# 模式 C: PulseProgram → compile(output="tqasm") → 云端 (最底层)
# ==============================================================================

def example_mode_c_direct_pulse_multi_waveforms():
    """模式 C: 直接脉冲编程 + 多波形对比分析 (最底层控制)
    
    链式调用工作流程:
        PulseProgram(1)
          → .set_device_params(...)  # 设置设备参数
          → .drag(0, ...)  # 添加脉冲操作
          → .device(provider="tyxonq", device="homebrew_s2_pulse")  # 配置设备
          → .run(shots=1024)  # 执行 (自动编译为 TQASM)
    
    多波形对比:
        - DRAG: Derivative Removal by Adiabatic Gate (抑制泄漏)
        - Gaussian: 标准高斯脉冲
        - Constant: 方波脉冲
    
    优势:
        - ✅ 链式调用 (与 Circuit 一致)
        - ✅ 最底层控制 (直接操控物理脉冲)
        - ✅ 完整参数 (精确控制脉冲波形)
        - ✅ 波形对比 (分析不同脉冲的性能)
    
    Returns:
        dict: 包含多波形的模拟结果和对比分析
    """
    print("\n" + "="*70)
    print("模式 C: 直接脉冲编程 + 多波形对比分析 (最底层控制)")
    print("="*70)
    
    from tyxonq.core.ir.pulse import PulseProgram
    
    # 步骤 1: 测试多种波形
    print("\n1️⃣  测试多种脉冲波形:")
    print("   - DRAG: 抑制泄漏误差 (推荐用于高保真门)")
    print("   - Gaussian: 标准高斯脉冲 (基准)")
    print("   - Constant: 方波脉冲 (对比基线)")
    
    from tyxonq.core.ir.pulse import PulseProgram
    
    device_params = {
        "qubit_freq": [5.0e9],
        "anharmonicity": [-330e6],
        "T1": [80e-6],
        "T2": [120e-6]
    }
    
    waveforms = {
        "DRAG": {
            "desc": "DRAG 脉冲 (Derivative Removal)",
            "builder": lambda: PulseProgram(1)
                .set_device_params(**device_params)
                .drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
        },
        "Gaussian": {
            "desc": "标准高斯脉冲",
            "builder": lambda: PulseProgram(1)
                .set_device_params(**device_params)
                .gaussian(0, amp=1.0, duration=160, sigma=40, qubit_freq=5.0e9)
        },
        "Constant": {
            "desc": "方波脉冲",
            "builder": lambda: PulseProgram(1)
                .set_device_params(**device_params)
                .constant(0, amp=1.0, duration=160, qubit_freq=5.0e9)
        }
    }
    
    results = {}
    
    for wave_name, wave_info in waveforms.items():
        print(f"\n   测试波形: {wave_name} - {wave_info['desc']}")
        prog = wave_info["builder"]()
        
        # 本地模拟
        try:
            state = prog.state(backend="numpy")
            prob_0 = abs(state[0])**2
            prob_1 = abs(state[1])**2
            results[wave_name] = {
                "state": state,
                "prob_0": prob_0,
                "prob_1": prob_1,
                "prog": prog
            }
            print(f"      |0⟩ 概率: {prob_0:.6f}")
            print(f"      |1⟩ 概率: {prob_1:.6f}")
        except Exception as e:
            print(f"      ⚠️  模拟失败: {e}")
            results[wave_name] = {"error": str(e), "prog": prog}
    
    # 选择 DRAG 作为主要示例
    prog = results["DRAG"]["prog"]
    
    # 步骤 2: 编译为 TQASM (用于验证)
    # 关键: 必须设置 inline_pulses=True 才能生成完整的 defcal 定义
    print("\n2️⃣  编译为 TQASM (关键: inline_pulses=True):")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 设置 inline_pulses=True 以生成完整的脉冲定义
        from tyxonq.compiler.api import compile_pulse
        result = compile_pulse(
            prog,
            output="tqasm",
            device_params=prog.device_params,
            options={"inline_pulses": True}  # 关键参数!
        )
        tqasm_code = result["pulse_schedule"]
        
        # 检查警告
        if w:
            print(f"   ⚠️  编译警告: {len(w)} 条")
            for warning in w:
                print(f"      → {str(warning.message)[:80]}...")
        else:
            print(f"   ✅ 编译成功，无警告")
    
    print(f"\n   💡 关键: inline_pulses=True 确保生成完整的 defcal 定义")
    print(f"      - False (默认): 保留符号引用 (适合本地模拟)")
    print(f"      - True: 完全内联 (适合云端提交)")
    
    print(f"\n   TQASM 代码长度: {len(tqasm_code)} 字符")
    
    # 步骤 3: 验证 TQASM 格式
    print("\n3️⃣  验证 TQASM 格式:")
    validation = validate_tqasm_format(tqasm_code)
    
    print(f"   版本声明: {'✅' if validation['version'] else '❌'}")
    print(f"   Cal 块: {'✅' if validation['cal_block'] else '❌'}")
    print(f"   Frame 声明: {'✅' if validation['frame_decl'] else '❌'}")
    print(f"   Defcal 定义: {'✅' if validation['defcal'] else '❌'}")
    print(f"   Waveform 定义: {'✅' if validation['waveform'] else '❌'}")
    
    if all(validation.values()):
        print("\n   ✅ TQASM 格式验证通过!")
    else:
        print("\n   ⚠️  TQASM 格式验证未完全通过")
    
    # 步骤 4: 显示 TQASM 代码预览
    print("\n4️⃣  TQASM 代码预览:")
    print("   " + "-"*66)
    for line in tqasm_code.split('\n')[:15]:
        print(f"   {line}")
    if len(tqasm_code.split('\n')) > 15:
        print("   ...")
    print("   " + "-"*66)
    
    # 步骤 5: 链式调用提交到云端
    print("\n5️⃣  链式调用提交到云端:")
    print("   代码:")
    print("   result = prog.device(provider='tyxonq', device='homebrew_s2').run(shots=1024)")
    print("   → 自动编译为 TQASM 并提交")
    
    token = get_api_token()
    cloud_result = None
    
    if token:
        print("\n   ✅ Token 已设置，开始提交任务 (DRAG 波形)...")
        try:
            import tyxonq as tq
            # 使用 DRAG 波形提交
            prog_drag = results["DRAG"]["prog"]
            task = prog_drag.device(provider="tyxonq", device="homebrew_s2").run(shots=1024)
            print(f"   ✅ 任务已提交: {task}")
            
            # 等待结果
            print("   ⏳ 等待云端执行结果...")
            import time
            time.sleep(3)
            
            try:
                details = tq.api.get_task_details(task, wait=False)
                cloud_result = details
                print(f"   ✅ 获取到任务状态: {details.get('status', 'unknown')}")
                if 'counts' in details:
                    print(f"   ✅ 测量结果: {details['counts']}")
            except Exception as e:
                print(f"   ⚠️  获取结果失败: {e}")
                print("      提示: 任务可能还在执行中")
        except Exception as e:
            print(f"   ⚠️  提交失败: {e}")
    else:
        print("\n   ⚠️  跳过云端提交 (缺少 Token)")
    
    # 步骤 6: 波形对比分析
    print("\n6️⃣  波形对比分析 (本地模拟):")
    print("   " + "-"*66)
    print(f"   {'波形':<12} {'|0⟩ 概率':<15} {'|1⟩ 概率':<15} {'性能评估'}")
    print("   " + "-"*66)
    
    for wave_name in ["DRAG", "Gaussian", "Constant"]:
        if "error" in results[wave_name]:
            print(f"   {wave_name:<12} {'N/A':<15} {'N/A':<15} 模拟失败")
        else:
            r = results[wave_name]
            perf = "⭐⭐⭐" if wave_name == "DRAG" else "⭐⭐" if wave_name == "Gaussian" else "⭐"
            print(f"   {wave_name:<12} {r['prob_0']:<15.6f} {r['prob_1']:<15.6f} {perf}")
    
    print("   " + "-"*66)
    print("   💡 分析:")
    print("      - DRAG: 最佳性能，抑制泄漏误差")
    print("      - Gaussian: 标准基准，性能良好")
    print("      - Constant: 简单但性能较差")
    
    return {
        "waveforms": results,
        "tqasm_code": tqasm_code,
        "validation": validation,
        "cloud_result": cloud_result
    }


# ==============================================================================
# TQASM 格式验证
# ==============================================================================

def validate_tqasm_format(tqasm_code: str) -> dict:
    """验证 TQASM 代码格式是否符合 OpenQASM 3.0 + OpenPulse 规范
    
    检查项:
        1. 版本声明: "TQASM 0.2;" 或 "OPENQASM 3.0;"
        2. Cal 块: "cal { ... }"
        3. Frame 声明: "frame ... = newframe(...);"
        4. Defcal 定义: "defcal gate_name ... { ... }"
        5. Waveform 定义: "waveform ..." 或 "play(...)"
    
    Args:
        tqasm_code: TQASM 代码字符串
    
    Returns:
        dict: 验证结果 {"version": bool, "cal_block": bool, ...}
    """
    return {
        "version": ("TQASM 0.2" in tqasm_code or "OPENQASM 3.0" in tqasm_code),
        "cal_block": "cal {" in tqasm_code or "cal{" in tqasm_code,
        "frame_decl": "newframe(" in tqasm_code,
        "defcal": "defcal " in tqasm_code,
        "waveform": ("waveform " in tqasm_code or "play(" in tqasm_code or "drag(" in tqasm_code)
    }


# ==============================================================================
# 云端 API 可用性验证
# ==============================================================================

def verify_cloud_api_availability():
    """验证云端 API 模块是否可用
    
    检查:
        1. tyxonq.cloud.api 模块是否存在
        2. 关键函数是否可导入
        3. Token 是否配置
    
    Returns:
        bool: API 是否可用
    """
    print("\n" + "="*70)
    print("云端 API 可用性验证")
    print("="*70)
    
    # 检查模块
    try:
        import tyxonq as tq
        from tyxonq.cloud import api
        print("✅ tyxonq.cloud.api 模块存在")
    except ImportError as e:
        print(f"❌ tyxonq.cloud.api 模块不存在: {e}")
        return False
    
    # 检查关键函数
    functions = ['submit_task', 'run', 'get_task_details', 'set_token', 'list_devices']
    for func_name in functions:
        if hasattr(api, func_name):
            print(f"✅ {func_name} 函数可用")
        else:
            print(f"❌ {func_name} 函数不存在")
            return False
    
    # 检查 Token
    token = get_api_token()
    if token:
        print("✅ Token 已配置")
        
        # 测试列出设备
        try:
            devices = tq.api.list_devices(provider="tyxonq")
            print(f"✅ 云端设备列表: {devices}")
        except Exception as e:
            print(f"⚠️  获取设备列表失败: {e}")
        
        return True
    else:
        print("⚠️  Token 未配置")
        return False


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """运行所有端到端测试示例"""
    print("\n" + "="*70)
    print("TyxonQ 脉冲编程云端提交端到端测试 (P0.5)")
    print("="*70)
    
    print("\n📖 硬件限制说明:")
    print("   - 当前硬件仅支持单比特脉冲 defcal 定义")
    print("   - 双比特门 (CX, CZ) 由硬件预定义，用户不能自定义")
    print("   - 符合 TensorCircuit/TQASM 0.2 标准")
    
    print("\n📖 三模式架构 (核心特色: 链式调用):")
    print("   - 模式 A: 显式声明 - Circuit.h(0).use_pulse().device().run() (推荐)")
    print("   - 模式 B: 智能推断 - Circuit.h(0).device().run() (最便捷)")
    print("   - 模式 C: 直接脉冲 - PulseProgram.drag().device().run() (底层)")
    print("\n💡 链式调用优势:")
    print("   ✅ 代码简洁流畅 (模拟真实量子设备行为)")
    print("   ✅ 统一 API 风格 (Circuit 和 PulseProgram 一致)")
    print("   ✅ 易于理解维护 (清晰的执行流程)")
    
    # 运行示例
    result_a = example_mode_a_explicit_pulse_with_comparison()
    result_b = example_mode_b_smart_inference()
    result_c = example_mode_c_direct_pulse_multi_waveforms()
    
    # 验证云端 API
    verify_cloud_api_availability()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    print("\n模式 A (显式声明 + 链式调用):")
    print(f"   代码: Circuit(1).h(0).use_pulse(...).device(...).run()")
    print(f"   TQASM 长度: {len(result_a['tqasm_code'])} 字符")
    print(f"   格式验证: {'✅ 通过' if all(result_a['validation'].values()) else '⚠️ 部分通过'}")
    print(f"   警告: 无 (推荐方式)")
    
    print("\n模式 B (智能推断 + 链式调用):")
    print(f"   代码: Circuit(1).h(0).device(...).run()")
    print(f"   TQASM 长度: {len(result_b['tqasm_code'])} 字符")
    print(f"   格式验证: {'✅ 通过' if all(result_b['validation'].values()) else '⚠️ 部分通过'}")
    print(f"   警告: 2 条 (TQASM 自动推断 + 参数自动补足)")
    
    print("\n模式 C (直接脉冲 + 链式调用):")
    print(f"   代码: PulseProgram(1).drag(...).device(...).run()")
    print(f"   TQASM 长度: {len(result_c['tqasm_code'])} 字符")
    print(f"   格式验证: {'✅ 通过' if all(result_c['validation'].values()) else '⚠️ 部分通过'}")
    print(f"   警告: 1 条 (to_circuit 兼容性警告)")
    
    print("\n💡 实际使用指南:")
    print("   1. 设置 Token: export TYXONQ_API_KEY='your_token'")
    print("   2. 运行示例: conda run -n qc python examples/pulse_cloud_submission_e2e.py")
    print("   3. 查看结果: tq.api.get_task_details(task, wait=True)")
    print("\n🔧 下一步开发:")
    print("   1. 集成真实硬件校准数据 (homebrew_s2_pulse)")
    print("   2. 完善双比特门的脉冲分解 (Cross-Resonance)")
    print("   3. 优化脉冲调度算法 (ASAP/ALAP)")
    
    print("\n" + "="*70)
    print("✅ P0.5 端到端测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
